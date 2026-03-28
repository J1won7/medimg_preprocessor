from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import json
import os
import pickle
import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
try:
    import blosc2
except ModuleNotFoundError:
    blosc2 = None
try:
    import torch
    from torch.utils.data import Dataset, get_worker_info
except ModuleNotFoundError:
    torch = None

    class Dataset:
        pass

    def get_worker_info():
        return None

from .config import PreprocessingConfig
from .preprocessing import RunStage, TaskMode, TaskPreprocessedCase


MANIFEST_FILENAME = "preprocessing_manifest.json"
DEFAULT_STORAGE_FORMAT = "blosc2"
DEFAULT_MASK_LOCATION_SAMPLES = 10000


def _fail_validation(message: str) -> None:
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)


def _require_torch() -> None:
    if torch is None:
        _fail_validation("torch is required for dataset loading and tensor conversion")


def _require_blosc2() -> None:
    if blosc2 is None:
        _fail_validation("blosc2 is required for the default preprocessed storage format")


def _serialize_config(config: Optional[PreprocessingConfig]) -> Optional[dict]:
    if config is None:
        return None
    if not isinstance(config, PreprocessingConfig):
        _fail_validation(f"config must be a PreprocessingConfig, got {type(config).__name__}")
    return asdict(config)


def _normalize_configuration_plans(configurations: Optional[Dict[str, dict]]) -> Optional[dict]:
    if configurations is None:
        return None
    if not isinstance(configurations, dict):
        _fail_validation("configurations must be a dict when provided")
    normalized = {}
    for name, payload in configurations.items():
        if not isinstance(payload, dict):
            _fail_validation(f"configuration '{name}' must be a dict")
        patch_size = payload.get("patch_size")
        spacing = payload.get("spacing")
        median_shape = payload.get("median_shape")
        recommended_batch_size = payload.get("recommended_batch_size")
        normalized[str(name)] = {
            "patch_size": None if patch_size is None else [int(i) for i in patch_size],
            "spacing": None if spacing is None else [float(i) for i in spacing],
            "median_shape": None if median_shape is None else [int(i) for i in median_shape],
            "recommended_batch_size": None if recommended_batch_size is None else int(recommended_batch_size),
        }
    return normalized


def _build_split_mapping(
    identifiers: Sequence[str],
    *,
    val_ratio: float,
    split_seed: int,
    label: str,
) -> Optional[dict]:
    if val_ratio <= 0:
        return None
    identifiers = [str(i) for i in identifiers]
    if len(identifiers) < 2:
        _fail_validation(f"{label} requires at least 2 cases to create an automatic train/val split")
    if not (0 < val_ratio < 1):
        _fail_validation(f"{label} val_ratio must be in the open interval (0, 1), got {val_ratio}")
    rng = np.random.RandomState(int(split_seed))
    shuffled = list(identifiers)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * float(val_ratio))))
    val_count = min(val_count, len(shuffled) - 1)
    val_identifiers = sorted(shuffled[:val_count])
    train_identifiers = sorted(shuffled[val_count:])
    if len(train_identifiers) == 0 or len(val_identifiers) == 0:
        _fail_validation(f"{label} automatic split must produce non-empty train and val sets")
    return {
        "train": train_identifiers,
        "val": val_identifiers,
        "val_ratio": float(val_ratio),
        "split_seed": int(split_seed),
    }


def _validate_task_mode(task_mode: str) -> None:
    valid = {
        TaskMode.SEGMENTATION,
        TaskMode.PAIRED_GENERATIVE,
        TaskMode.UNPAIRED_GENERATIVE,
        TaskMode.SELF_SUPERVISED,
    }
    if task_mode not in valid:
        _fail_validation(f"Unsupported task_mode '{task_mode}'")


def _validate_run_stage(run_stage: str) -> None:
    valid = {RunStage.TRAIN, RunStage.PREDICT, RunStage.PREDICT_AND_EVALUATE}
    if run_stage not in valid:
        _fail_validation(f"Unsupported run_stage '{run_stage}'")


def _write_json(filename: str, payload: dict) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _read_json(filename: str) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        _fail_validation(f"Manifest file must contain a JSON object, got {type(payload).__name__}")
    return payload


def _sample_mask_locations(mask: np.ndarray, *, seed: int = 1234, max_samples: int = DEFAULT_MASK_LOCATION_SAMPLES) -> list[list[int]]:
    mask = np.asarray(mask).astype(bool)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return []
    rng = np.random.RandomState(int(seed))
    if len(coords) > int(max_samples):
        coords = coords[rng.choice(len(coords), int(max_samples), replace=False)]
    return [[int(i) for i in row] for row in coords]


def _validate_storage_format(storage_format: str) -> str:
    storage_format = str(storage_format).lower()
    if storage_format not in {"blosc2", "npz"}:
        _fail_validation(f"Unsupported storage format '{storage_format}'")
    return storage_format


def _comp_blosc2_params(
    image_size: Sequence[int],
    patch_size: Optional[Sequence[int]],
    bytes_per_pixel: int,
    l1_cache_size_per_core_in_bytes: int = 32768,
    l3_cache_size_per_core_in_bytes: int = 1441792,
    safety_factor: float = 0.8,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    image_size = tuple(int(i) for i in image_size)
    if patch_size is None:
        patch_size = image_size[1:]
    if len(patch_size) == 2:
        patch_size = [1, *patch_size]
    patch_size = np.array(patch_size)
    num_channels = image_size[0]
    block_size = np.array((num_channels, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size]))

    estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
    while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
        axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
        idx = 0
        picked_axis = axis_order[idx]
        while block_size[picked_axis + 1] == 1:
            idx += 1
            picked_axis = axis_order[idx]
        block_size[picked_axis + 1] = 2 ** (max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1))))
        block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

    block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])
    chunk_size = block_size.copy()
    estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
    while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
        if patch_size[0] == 1 and all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
            break
        if all([i == j for i, j in zip(chunk_size, image_size)]):
            break
        axis_order = np.argsort(chunk_size[1:] / block_size[1:])
        idx = 0
        picked_axis = axis_order[idx]
        while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
            idx += 1
            picked_axis = axis_order[idx]
        chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
        chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
            chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
            break
    chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]
    return tuple(block_size), tuple(chunk_size)


def _save_blosc2_array(array: np.ndarray, filename: str, patch_size: Optional[Sequence[int]]) -> None:
    _require_blosc2()
    blosc2.set_nthreads(1)
    blocks, chunks = _comp_blosc2_params(array.shape, patch_size, array.itemsize)
    cparams = {"codec": blosc2.Codec.ZSTD, "clevel": 8}
    blosc2.asarray(np.ascontiguousarray(array), urlpath=filename, chunks=chunks, blocks=blocks, cparams=cparams)


def save_preprocessed_case(
    case: TaskPreprocessedCase,
    output_filename_truncated: str,
    *,
    storage_format: str = DEFAULT_STORAGE_FORMAT,
    patch_size_hint: Optional[Sequence[int]] = None,
    patch_sampling_patch_sizes: Optional[Dict[str, Sequence[int]]] = None,
    patch_sampling_threshold: Optional[float] = None,
    patch_sampling_min_fraction: float = 0.0,
    patch_sampling_source: str = "image",
    patch_sampling_max_starts: int = 8192,
) -> None:
    if not isinstance(case, TaskPreprocessedCase):
        _fail_validation(f"case must be a TaskPreprocessedCase, got {type(case).__name__}")
    if not isinstance(case.image, np.ndarray):
        _fail_validation("case.image must be a numpy.ndarray")
    if not isinstance(case.properties, dict):
        _fail_validation("case.properties must be a dict")
    output_dir = os.path.dirname(output_filename_truncated) or "."
    if not os.path.isdir(output_dir):
        _fail_validation(f"Output directory does not exist: {output_dir}")
    if case.target is not None and not isinstance(case.target, np.ndarray):
        _fail_validation("case.target must be a numpy.ndarray when provided")
    if case.evaluation_reference is not None and not isinstance(case.evaluation_reference, np.ndarray):
        _fail_validation("case.evaluation_reference must be a numpy.ndarray when provided")
    if case.target is not None and tuple(case.target.shape[1:]) != tuple(case.image.shape[1:]):
        _fail_validation(
            f"case.target spatial shape must match image, got {case.target.shape[1:]} and {case.image.shape[1:]}"
        )
    if case.evaluation_reference is not None and tuple(case.evaluation_reference.shape[1:]) != tuple(case.image.shape[1:]):
        _fail_validation(
            "case.evaluation_reference spatial shape must match image, "
            f"got {case.evaluation_reference.shape[1:]} and {case.image.shape[1:]}"
        )
    stored_mask = case.mask if case.mask is not None else case.patch_sampling_mask
    if stored_mask is not None:
        stored_mask = np.asarray(stored_mask)
        if tuple(stored_mask.shape) != tuple(case.image.shape[1:]):
            _fail_validation(
                f"case.mask spatial shape must match image, got {stored_mask.shape} and {case.image.shape[1:]}"
            )
    storage_format = _validate_storage_format(storage_format)
    if storage_format == "npz":
        arrays = {"image": np.ascontiguousarray(case.image)}
        if case.target is not None:
            arrays["target"] = np.ascontiguousarray(case.target)
        if case.evaluation_reference is not None:
            arrays["evaluation_reference"] = np.ascontiguousarray(case.evaluation_reference)
        if stored_mask is not None:
            arrays["mask"] = np.ascontiguousarray(stored_mask.astype(np.uint8, copy=False))
        np.savez_compressed(output_filename_truncated + ".npz", **arrays)
    else:
        _save_blosc2_array(np.ascontiguousarray(case.image), output_filename_truncated + ".b2nd", patch_size_hint)
        if case.target is not None:
            _save_blosc2_array(
                np.ascontiguousarray(case.target),
                output_filename_truncated + "_target.b2nd",
                patch_size_hint,
            )
        if case.evaluation_reference is not None:
            _save_blosc2_array(
                np.ascontiguousarray(case.evaluation_reference),
                output_filename_truncated + "_evalref.b2nd",
                patch_size_hint,
            )
        if stored_mask is not None:
            _save_blosc2_array(
                np.ascontiguousarray(stored_mask[None].astype(np.uint8, copy=False)),
                output_filename_truncated + "_mask.b2nd",
                patch_size_hint,
            )
    metadata = {
        "properties": case.properties,
        "target_properties": case.target_properties,
        "evaluation_properties": case.evaluation_properties,
        "task_mode": case.task_mode,
        "run_stage": case.run_stage,
        "reference_type": case.reference_type,
        "storage_format": storage_format,
        "mask_locations": _sample_mask_locations(
            stored_mask,
            max_samples=int(patch_sampling_max_starts),
        )
        if stored_mask is not None
        else None,
        "patch_sampling": None,
    }
    with open(output_filename_truncated + ".pkl", "wb") as f:
        pickle.dump(metadata, f)


def save_preprocessed_dataset_manifest(
    folder: str,
    task_mode: str,
    run_stage: str = RunStage.TRAIN,
    *,
    config: Optional[PreprocessingConfig] = None,
    default_patch_size: Optional[Sequence[int]] = None,
    default_configuration: Optional[str] = None,
    configurations: Optional[Dict[str, dict]] = None,
    identifiers: Optional[Sequence[str]] = None,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    storage_format: str = DEFAULT_STORAGE_FORMAT,
) -> str:
    if not os.path.isdir(folder):
        _fail_validation(f"Preprocessed dataset folder does not exist: {folder}")
    _validate_task_mode(task_mode)
    _validate_run_stage(run_stage)
    if task_mode == TaskMode.UNPAIRED_GENERATIVE:
        _fail_validation("Use save_unpaired_preprocessed_dataset_manifest for unpaired_generative datasets")
    manifest_identifiers = list(identifiers) if identifiers is not None else _list_identifiers(folder)
    manifest = {
        "format_version": 1,
        "task_mode": task_mode,
        "run_stage": run_stage,
        "dataset_kind": "single_folder",
        "storage_format": _validate_storage_format(storage_format),
        "default_patch_size": None if default_patch_size is None else [int(i) for i in default_patch_size],
        "default_configuration": default_configuration,
        "configurations": _normalize_configuration_plans(configurations),
        "identifiers": manifest_identifiers,
        "splits": _build_split_mapping(
            manifest_identifiers,
            val_ratio=val_ratio,
            split_seed=split_seed,
            label=f"{task_mode} dataset",
        )
        if run_stage == RunStage.TRAIN
        else None,
        "preprocessing_config": _serialize_config(config),
    }
    manifest_file = os.path.join(folder, MANIFEST_FILENAME)
    _write_json(manifest_file, manifest)
    return manifest_file


def save_preprocessed_dataset(
    folder: str,
    task_mode: str,
    run_stage: str = RunStage.TRAIN,
    *,
    config: Optional[PreprocessingConfig] = None,
    default_patch_size: Optional[Sequence[int]] = None,
    default_configuration: Optional[str] = None,
    configurations: Optional[Dict[str, dict]] = None,
    identifiers: Optional[Sequence[str]] = None,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    folder_a: Optional[str] = None,
    folder_b: Optional[str] = None,
    config_a: Optional[PreprocessingConfig] = None,
    config_b: Optional[PreprocessingConfig] = None,
    identifiers_a: Optional[Sequence[str]] = None,
    identifiers_b: Optional[Sequence[str]] = None,
    random_pairing: bool = True,
    storage_format: str = DEFAULT_STORAGE_FORMAT,
) -> str:
    _validate_task_mode(task_mode)
    storage_format = _validate_storage_format(storage_format)
    if task_mode == TaskMode.UNPAIRED_GENERATIVE:
        if folder_a is None or folder_b is None:
            _fail_validation("unpaired_generative save_preprocessed_dataset requires folder_a and folder_b")
        return save_unpaired_preprocessed_dataset_manifest(
            root_folder=folder,
            folder_a=folder_a,
            folder_b=folder_b,
            run_stage=run_stage,
            config_a=config_a,
            config_b=config_b,
            default_patch_size=default_patch_size,
            default_configuration=default_configuration,
            configurations=configurations,
            identifiers_a=identifiers_a,
            identifiers_b=identifiers_b,
            val_ratio=val_ratio,
            split_seed=split_seed,
            random_pairing=random_pairing,
            storage_format=storage_format,
        )
    if folder_a is not None or folder_b is not None:
        _fail_validation(f"{task_mode} save_preprocessed_dataset does not accept folder_a/folder_b")
    if config_a is not None or config_b is not None:
        _fail_validation(f"{task_mode} save_preprocessed_dataset does not accept config_a/config_b")
    if identifiers_a is not None or identifiers_b is not None:
        _fail_validation(f"{task_mode} save_preprocessed_dataset does not accept identifiers_a/identifiers_b")
    return save_preprocessed_dataset_manifest(
        folder=folder,
        task_mode=task_mode,
        run_stage=run_stage,
        config=config,
        default_patch_size=default_patch_size,
        default_configuration=default_configuration,
        configurations=configurations,
        identifiers=identifiers,
        val_ratio=val_ratio,
        split_seed=split_seed,
        storage_format=storage_format,
    )


def save_unpaired_preprocessed_dataset_manifest(
    root_folder: str,
    folder_a: str,
    folder_b: str,
    run_stage: str = RunStage.TRAIN,
    *,
    config_a: Optional[PreprocessingConfig] = None,
    config_b: Optional[PreprocessingConfig] = None,
    default_patch_size: Optional[Sequence[int]] = None,
    default_configuration: Optional[str] = None,
    configurations: Optional[Dict[str, dict]] = None,
    identifiers_a: Optional[Sequence[str]] = None,
    identifiers_b: Optional[Sequence[str]] = None,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    random_pairing: bool = True,
    storage_format: str = DEFAULT_STORAGE_FORMAT,
) -> str:
    if not os.path.isdir(root_folder):
        _fail_validation(f"Preprocessed dataset root folder does not exist: {root_folder}")
    _validate_run_stage(run_stage)
    if run_stage == RunStage.PREDICT_AND_EVALUATE:
        _fail_validation("unpaired_generative does not support predict_and_evaluate manifests")
    folder_a_abs = folder_a if os.path.isabs(folder_a) else os.path.join(root_folder, folder_a)
    folder_b_abs = folder_b if os.path.isabs(folder_b) else os.path.join(root_folder, folder_b)
    if not os.path.isdir(folder_a_abs):
        _fail_validation(f"Domain A folder does not exist: {folder_a_abs}")
    if not os.path.isdir(folder_b_abs):
        _fail_validation(f"Domain B folder does not exist: {folder_b_abs}")
    manifest_identifiers_a = list(identifiers_a) if identifiers_a is not None else _list_identifiers(folder_a_abs)
    manifest_identifiers_b = list(identifiers_b) if identifiers_b is not None else _list_identifiers(folder_b_abs)
    manifest = {
        "format_version": 1,
        "task_mode": TaskMode.UNPAIRED_GENERATIVE,
        "run_stage": run_stage,
        "dataset_kind": "unpaired_domains",
        "storage_format": _validate_storage_format(storage_format),
        "default_patch_size": None if default_patch_size is None else [int(i) for i in default_patch_size],
        "default_configuration": default_configuration,
        "configurations": _normalize_configuration_plans(configurations),
        "random_pairing": bool(random_pairing),
        "domains": {
            "a": {
                "folder": os.path.relpath(folder_a_abs, root_folder),
                "identifiers": manifest_identifiers_a,
                "splits": _build_split_mapping(
                    manifest_identifiers_a,
                    val_ratio=val_ratio,
                    split_seed=split_seed,
                    label="unpaired domain A",
                )
                if run_stage == RunStage.TRAIN
                else None,
                "preprocessing_config": _serialize_config(config_a),
            },
            "b": {
                "folder": os.path.relpath(folder_b_abs, root_folder),
                "identifiers": manifest_identifiers_b,
                "splits": _build_split_mapping(
                    manifest_identifiers_b,
                    val_ratio=val_ratio,
                    split_seed=split_seed,
                    label="unpaired domain B",
                )
                if run_stage == RunStage.TRAIN
                else None,
                "preprocessing_config": _serialize_config(config_b),
            },
        },
    }
    manifest_file = os.path.join(root_folder, MANIFEST_FILENAME)
    _write_json(manifest_file, manifest)
    return manifest_file


def load_preprocessed_dataset_manifest(folder: str) -> dict:
    if not os.path.isdir(folder):
        _fail_validation(f"Preprocessed dataset folder does not exist: {folder}")
    manifest_file = os.path.join(folder, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_file):
        _fail_validation(f"Preprocessed dataset manifest does not exist: {manifest_file}")
    manifest = _read_json(manifest_file)
    if manifest.get("format_version") != 1:
        _fail_validation(f"Unsupported manifest format_version '{manifest.get('format_version')}'")
    task_mode = manifest.get("task_mode")
    run_stage = manifest.get("run_stage")
    dataset_kind = manifest.get("dataset_kind")
    if not isinstance(task_mode, str):
        _fail_validation("Manifest must contain a string 'task_mode'")
    if not isinstance(run_stage, str):
        _fail_validation("Manifest must contain a string 'run_stage'")
    _validate_task_mode(task_mode)
    _validate_run_stage(run_stage)
    if dataset_kind not in {"single_folder", "unpaired_domains"}:
        _fail_validation(f"Unsupported dataset_kind '{dataset_kind}' in manifest")
    storage_format = manifest.get("storage_format", DEFAULT_STORAGE_FORMAT)
    _validate_storage_format(storage_format)
    if task_mode == TaskMode.UNPAIRED_GENERATIVE and dataset_kind != "unpaired_domains":
        _fail_validation("unpaired_generative manifest must use dataset_kind 'unpaired_domains'")
    if task_mode != TaskMode.UNPAIRED_GENERATIVE and dataset_kind != "single_folder":
        _fail_validation(f"{task_mode} manifest must use dataset_kind 'single_folder'")
    if manifest.get("default_patch_size") is not None and not isinstance(manifest["default_patch_size"], list):
        _fail_validation("Manifest 'default_patch_size' must be a list when provided")
    if manifest.get("default_configuration") is not None and not isinstance(manifest["default_configuration"], str):
        _fail_validation("Manifest 'default_configuration' must be a string when provided")
    if manifest.get("splits") is not None:
        if not isinstance(manifest["splits"], dict):
            _fail_validation("Manifest 'splits' must be a dict when provided")
        for split_name in ("train", "val"):
            if split_name not in manifest["splits"] or not isinstance(manifest["splits"][split_name], list):
                _fail_validation(f"Manifest 'splits' must contain a list '{split_name}'")
    if manifest.get("configurations") is not None:
        if not isinstance(manifest["configurations"], dict):
            _fail_validation("Manifest 'configurations' must be a dict when provided")
        for name, payload in manifest["configurations"].items():
            if not isinstance(payload, dict):
                _fail_validation(f"Manifest configuration '{name}' must be a dict")
            for key in ("patch_size", "spacing", "median_shape"):
                if payload.get(key) is not None and not isinstance(payload[key], list):
                    _fail_validation(f"Manifest configuration '{name}' field '{key}' must be a list when provided")
            if payload.get("recommended_batch_size") is not None and not isinstance(payload["recommended_batch_size"], int):
                _fail_validation(
                    f"Manifest configuration '{name}' field 'recommended_batch_size' must be an int when provided"
                )
    if dataset_kind == "single_folder":
        identifiers = manifest.get("identifiers")
        if identifiers is not None and not isinstance(identifiers, list):
            _fail_validation("Manifest 'identifiers' must be a list when provided")
    if dataset_kind == "unpaired_domains":
        domains = manifest.get("domains")
        if not isinstance(domains, dict):
            _fail_validation("Unpaired manifest must contain a 'domains' mapping")
        for domain_key in ("a", "b"):
            if domain_key not in domains or not isinstance(domains[domain_key], dict):
                _fail_validation(f"Unpaired manifest must contain a '{domain_key}' domain mapping")
            if not isinstance(domains[domain_key].get("folder"), str):
                _fail_validation(f"Unpaired manifest domain '{domain_key}' must contain a string 'folder'")
            identifiers = domains[domain_key].get("identifiers")
            if identifiers is not None and not isinstance(identifiers, list):
                _fail_validation(
                    f"Unpaired manifest domain '{domain_key}' must contain a list 'identifiers' when provided"
                )
            splits = domains[domain_key].get("splits")
            if splits is not None:
                if not isinstance(splits, dict):
                    _fail_validation(f"Unpaired manifest domain '{domain_key}' must contain a dict 'splits'")
                for split_name in ("train", "val"):
                    if split_name not in splits or not isinstance(splits[split_name], list):
                        _fail_validation(
                            f"Unpaired manifest domain '{domain_key}' splits must contain a list '{split_name}'"
                        )
    return manifest


def _load_conflict_map(extra_folder: str | None, identifier: str, storage_format: str):
    if extra_folder is None:
        return None
    b2nd_conflict_file = os.path.join(extra_folder, identifier + "_conflict.b2nd")
    npy_conflict_file = os.path.join(extra_folder, identifier + "_conflict.npy")
    npz_conflict_file = os.path.join(extra_folder, identifier + "_conflict.npz")

    if storage_format == "blosc2" and os.path.isfile(b2nd_conflict_file):
        _require_blosc2()
        blosc2.set_nthreads(1)
        dparams = {"nthreads": 1}
        mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
        return blosc2.open(urlpath=b2nd_conflict_file, mode="r", dparams=dparams, **mmap_kwargs)
    if os.path.isfile(npy_conflict_file):
        return np.load(npy_conflict_file, mmap_mode="r")
    if os.path.isfile(npz_conflict_file):
        with np.load(npz_conflict_file) as arrays:
            if "conflict_map" not in arrays.files:
                _fail_validation(
                    f"Conflict map file for case '{identifier}' must contain a 'conflict_map' array"
                )
            return arrays["conflict_map"]
    return None


@lru_cache(maxsize=2)
def _load_preprocessed_case_cached(folder: str, identifier: str, extra_folder: str | None = None) -> dict:
    if not os.path.isdir(folder):
        _fail_validation(f"Preprocessed case folder does not exist: {folder}")
    b2nd_image_file = os.path.join(folder, identifier + ".b2nd")
    b2nd_target_file = os.path.join(folder, identifier + "_target.b2nd")
    b2nd_eval_file = os.path.join(folder, identifier + "_evalref.b2nd")
    b2nd_mask_file = os.path.join(folder, identifier + "_mask.b2nd")
    npz_file = os.path.join(folder, identifier + ".npz")
    pkl_file = os.path.join(folder, identifier + ".pkl")
    has_blosc2 = os.path.isfile(b2nd_image_file)
    has_npz = os.path.isfile(npz_file)
    if not has_blosc2 and not has_npz:
        _fail_validation(f"Preprocessed array file does not exist for case '{identifier}' in {folder}")
    if not os.path.isfile(pkl_file):
        _fail_validation(f"Preprocessed metadata file does not exist: {pkl_file}")
    with open(pkl_file, "rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, dict):
        _fail_validation(f"Metadata for case '{identifier}' must be a dict")
    storage_format = _validate_storage_format(meta.get("storage_format", "blosc2" if has_blosc2 else "npz"))
    if storage_format == "blosc2":
        _require_blosc2()
        blosc2.set_nthreads(1)
        dparams = {"nthreads": 1}
        mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
        image = blosc2.open(urlpath=b2nd_image_file, mode="r", dparams=dparams, **mmap_kwargs)
        target = blosc2.open(urlpath=b2nd_target_file, mode="r", dparams=dparams, **mmap_kwargs) if os.path.isfile(b2nd_target_file) else None
        evaluation_reference = (
            blosc2.open(urlpath=b2nd_eval_file, mode="r", dparams=dparams, **mmap_kwargs)
            if os.path.isfile(b2nd_eval_file)
            else None
        )
        mask = blosc2.open(urlpath=b2nd_mask_file, mode="r", dparams=dparams, **mmap_kwargs) if os.path.isfile(b2nd_mask_file) else None
        conflict_map = _load_conflict_map(extra_folder, identifier, storage_format)
        case = {
            "image": image,
            "target": target,
            "evaluation_reference": evaluation_reference,
            "mask": None if mask is None else mask[0],
            "conflict_map": None if conflict_map is None else conflict_map,
            **meta,
        }
    else:
        with np.load(npz_file) as arrays:
            conflict_map = _load_conflict_map(extra_folder, identifier, storage_format)
            if "image" not in arrays.files:
                _fail_validation(f"Preprocessed case '{identifier}' does not contain an 'image' array")
            case = {
                "image": arrays["image"],
                "target": arrays["target"] if "target" in arrays.files else None,
                "evaluation_reference": arrays["evaluation_reference"]
                if "evaluation_reference" in arrays.files
                else None,
                "mask": arrays["mask"] if "mask" in arrays.files else None,
                "conflict_map": None if conflict_map is None else conflict_map,
                **meta,
            }
    if "properties" not in case or not isinstance(case["properties"], dict):
        _fail_validation(f"Metadata for case '{identifier}' must contain a dict 'properties'")
    if case["target"] is not None and tuple(case["target"].shape[1:]) != tuple(case["image"].shape[1:]):
        _fail_validation(
            f"Preprocessed case '{identifier}' has mismatched image/target shapes: "
            f"{case['image'].shape[1:]} vs {case['target'].shape[1:]}"
        )
    if (
        case["evaluation_reference"] is not None
        and tuple(case["evaluation_reference"].shape[1:]) != tuple(case["image"].shape[1:])
    ):
        _fail_validation(
            f"Preprocessed case '{identifier}' has mismatched image/evaluation_reference shapes: "
            f"{case['image'].shape[1:]} vs {case['evaluation_reference'].shape[1:]}"
        )
    if case.get("mask") is not None and tuple(case["mask"].shape) != tuple(case["image"].shape[1:]):
        _fail_validation(
            f"Preprocessed case '{identifier}' has mismatched image/mask shapes: "
            f"{case['image'].shape[1:]} vs {case['mask'].shape}"
        )
    if case.get("conflict_map") is not None:
        conflict_map = case["conflict_map"]
        expected_shape = tuple(case["image"].shape)
        expected_spatial = tuple(case["image"].shape[1:])
        if tuple(conflict_map.shape) == expected_spatial:
            case["conflict_map"] = np.asarray(conflict_map)[None]
            conflict_map = case["conflict_map"]
        if tuple(conflict_map.shape) != expected_shape:
            _fail_validation(
                f"Preprocessed case '{identifier}' has mismatched image/conflict_map shapes: "
                f"{expected_shape} vs {tuple(conflict_map.shape)}"
            )
    if case.get("mask_locations") is None and case.get("mask") is not None:
        case["mask_locations"] = _sample_mask_locations(np.asarray(case["mask"]))
    return case


def load_preprocessed_case(folder: str, identifier: str, extra_folder: str | None = None) -> dict:
    normalized_extra = None if extra_folder is None else str(extra_folder)
    return dict(_load_preprocessed_case_cached(str(folder), str(identifier), normalized_extra))


def save_preprocessed_conflict_map(
    folder: str,
    identifier: str,
    conflict_map: np.ndarray,
    *,
    extra_folder: str | None = None,
    patch_size_hint: Optional[Sequence[int]] = None,
) -> str:
    case = load_preprocessed_case(folder, identifier)
    array = np.asarray(conflict_map, dtype=np.float32)
    if tuple(array.shape) == tuple(case["image"].shape[1:]):
        array = array[None]
    expected_shape = tuple(case["image"].shape)
    if tuple(array.shape) != expected_shape:
        _fail_validation(
            f"conflict_map for case '{identifier}' must match image shape {expected_shape}, got {tuple(array.shape)}"
        )
    target_folder = extra_folder if extra_folder is not None else folder
    os.makedirs(target_folder, exist_ok=True)
    storage_format = case.get("storage_format", DEFAULT_STORAGE_FORMAT)
    if storage_format == "blosc2":
        output_path = os.path.join(target_folder, identifier + "_conflict.b2nd")
        _save_blosc2_array(np.ascontiguousarray(array), output_path, patch_size_hint)
    else:
        output_path = os.path.join(target_folder, identifier + "_conflict.npy")
        np.save(output_path, np.ascontiguousarray(array))
    return output_path


def _list_identifiers(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        _fail_validation(f"Preprocessed dataset folder does not exist: {folder}")
    identifiers = set()
    for name in os.listdir(folder):
        if name.endswith(".npz"):
            identifiers.add(os.path.splitext(name)[0])
        elif (
            name.endswith(".b2nd")
            and not name.endswith("_target.b2nd")
            and not name.endswith("_evalref.b2nd")
            and not name.endswith("_mask.b2nd")
        ):
            identifiers.add(name[:-5])
    return sorted(identifiers)


def _validate_patch_dims(array: np.ndarray, patch_size: Optional[Sequence[int]], context: str) -> None:
    if patch_size is None:
        return
    spatial_dims = array.ndim - 1
    if len(patch_size) == spatial_dims:
        return
    if spatial_dims == 3 and len(patch_size) == 2:
        return
    _fail_validation(
        f"{context} expects patch_size with {spatial_dims} spatial dims, got {len(patch_size)} "
        f"for array shape {array.shape}"
    )


def _resolve_patch_size(array: np.ndarray, patch_size: Optional[Sequence[int]], context: str) -> Optional[tuple[int, ...]]:
    if patch_size is None:
        return None
    spatial_dims = array.ndim - 1
    _validate_patch_dims(array, patch_size, context)
    if len(patch_size) == spatial_dims:
        return tuple(int(i) for i in patch_size)
    if spatial_dims == 3 and len(patch_size) == 2:
        return (1, int(patch_size[0]), int(patch_size[1]))
    _fail_validation(
        f"{context} expects patch_size with {spatial_dims} spatial dims, got {len(patch_size)} "
        f"for array shape {array.shape}"
    )


def _resolve_patch_size_from_spatial_shape(
    spatial_shape: Sequence[int],
    patch_size: Optional[Sequence[int]],
    context: str,
) -> Optional[tuple[int, ...]]:
    if patch_size is None:
        return None
    spatial_dims = len(tuple(int(i) for i in spatial_shape))
    if len(patch_size) == spatial_dims:
        return tuple(int(i) for i in patch_size)
    if spatial_dims == 3 and len(patch_size) == 2:
        return (1, int(patch_size[0]), int(patch_size[1]))
    _fail_validation(
        f"{context} expects patch_size with {spatial_dims} spatial dims, got {len(patch_size)} "
        f"for spatial shape {tuple(int(i) for i in spatial_shape)}"
    )


def _build_runtime_rng(seed: int, index: int) -> np.random.RandomState:
    base = int(seed) + int(index) * 1009
    worker_info = get_worker_info()
    if worker_info is not None:
        base += int(worker_info.id) * 104729
    base += int(np.random.randint(0, np.iinfo(np.int32).max))
    return np.random.RandomState(base % np.iinfo(np.int32).max)


def _crop_spatial(array: np.ndarray, target: Sequence[int], starts: Sequence[int]) -> np.ndarray:
    slicer = (slice(None),) + tuple(slice(s, s + p) for s, p in zip(starts, target))
    return array[slicer]


def _squeeze_2d_patch_if_needed(array: np.ndarray, patch_size: Optional[Sequence[int]]) -> np.ndarray:
    if patch_size is None:
        return array
    if array.ndim == 4 and len(patch_size) == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array


def _crop_or_pad(array: np.ndarray, patch_size: Optional[Sequence[int]], rng: np.random.RandomState) -> np.ndarray:
    if patch_size is None:
        return array
    target = _resolve_patch_size(array, patch_size, "crop/pad")
    pad_width = [(0, 0)]
    for current, wanted in zip(array.shape[1:], target):
        missing = max(wanted - current, 0)
        before = missing // 2
        after = missing - before
        pad_width.append((before, after))
    if any(p != (0, 0) for p in pad_width):
        array = np.pad(array, pad_width, mode="constant", constant_values=0)
    starts = []
    for current, wanted in zip(array.shape[1:], target):
        starts.append(0 if current == wanted else int(rng.randint(0, current - wanted + 1)))
    return _squeeze_2d_patch_if_needed(_crop_spatial(array, target, starts), patch_size)


def _pad_to_patch_size(array: np.ndarray, patch_size: Sequence[int]) -> np.ndarray:
    target = _resolve_patch_size(array, patch_size, "crop/pad")
    pad_width = [(0, 0)]
    for current, wanted in zip(array.shape[1:], target):
        missing = max(wanted - current, 0)
        before = missing // 2
        after = missing - before
        pad_width.append((before, after))
    if any(p != (0, 0) for p in pad_width):
        array = np.pad(array, pad_width, mode="constant", constant_values=0)
    return array


def _compute_crop_starts(
    spatial_shape: Sequence[int],
    patch_size: Sequence[int],
    rng: np.random.RandomState,
) -> Tuple[int, ...]:
    effective_patch = tuple(int(i) for i in patch_size)
    if len(spatial_shape) == 3 and len(effective_patch) == 2:
        effective_patch = (1, *effective_patch)
    if len(spatial_shape) != len(effective_patch):
        _fail_validation(
            f"crop/pad expects patch_size with {len(spatial_shape)} spatial dims, got {len(patch_size)} "
            f"for spatial shape {tuple(spatial_shape)}"
        )
    starts = []
    for current, wanted in zip(spatial_shape, effective_patch):
        current = max(current, wanted)
        starts.append(0 if current == wanted else int(rng.randint(0, current - wanted + 1)))
    return tuple(starts)


def _crop_with_starts(array: np.ndarray, patch_size: Sequence[int], starts: Sequence[int]) -> np.ndarray:
    target = _resolve_patch_size(array, patch_size, "crop/pad")
    if len(starts) != len(target):
        _fail_validation(f"crop/pad received {len(starts)} crop starts for patch_size with {len(target)} dims")
    array = _pad_to_patch_size(array, target)
    return _squeeze_2d_patch_if_needed(_crop_spatial(array, target, starts), patch_size)


def _patch_key(patch_size: Sequence[int]) -> str:
    return "x".join(str(int(i)) for i in patch_size)


def _integral_sum_nd(integral: np.ndarray, starts: Sequence[int], patch_size: Sequence[int]) -> int:
    dims = len(patch_size)
    ends = [min(int(s) + int(p) - 1, int(integral.shape[axis]) - 1) for axis, (s, p) in enumerate(zip(starts, patch_size))]
    total = 0.0
    for mask_bits in range(1 << dims):
        index = []
        sign = 1
        for axis in range(dims):
            if mask_bits & (1 << axis):
                index.append(int(starts[axis]) - 1)
                sign *= -1
            else:
                index.append(int(ends[axis]))
        if all(i >= 0 for i in index):
            total += sign * float(integral[tuple(index)])
    return int(total)


def _compute_patch_sampling_starts(
    source: np.ndarray,
    patch_size: Sequence[int],
    *,
    threshold: float,
    min_fraction: float,
    max_starts: int = 8192,
) -> list[list[int]]:
    if source.ndim >= 4:
        foreground = np.any(source > threshold, axis=0)
    else:
        foreground = source > threshold
    if not np.any(foreground):
        return []
    dummy = np.zeros((1, *foreground.shape), dtype=np.uint8)
    target = _resolve_patch_size(dummy, patch_size, "patch sampling")
    if target is None:
        return []
    target = tuple(int(i) for i in target)
    spatial_shape = tuple(int(i) for i in foreground.shape)
    if len(spatial_shape) != len(target):
        return []
    max_starts_per_axis = [max(dim - size, 0) for dim, size in zip(spatial_shape, target)]
    strides = [max(1, size // 4) for size in target]
    axes = [list(range(0, max_start + 1, stride)) for max_start, stride in zip(max_starts_per_axis, strides)]
    for axis, max_start in enumerate(max_starts_per_axis):
        if axes[axis][-1] != max_start:
            axes[axis].append(max_start)
    integral = foreground.astype(np.int32)
    for axis in range(integral.ndim):
        integral = integral.cumsum(axis=axis)
    effective_patch_voxels = float(np.prod([min(dim, size) for dim, size in zip(spatial_shape, target)]))
    min_count = effective_patch_voxels * float(min_fraction)
    valid_starts: list[list[int]] = []
    for starts_idx in np.ndindex(*(len(values) for values in axes)):
        starts = tuple(axes[axis][idx] for axis, idx in enumerate(starts_idx))
        if _integral_sum_nd(integral, starts, target) >= min_count:
            valid_starts.append([int(i) for i in starts])
            if len(valid_starts) >= int(max_starts):
                break
    return valid_starts


def _compute_patch_sampling_starts_from_mask(
    foreground: np.ndarray,
    patch_size: Sequence[int],
    *,
    min_fraction: float,
    max_starts: int = 8192,
) -> list[list[int]]:
    foreground = np.asarray(foreground).astype(bool)
    if foreground.ndim >= 4:
        foreground = np.any(foreground, axis=0)
    if not np.any(foreground):
        return []
    dummy = np.zeros((1, *foreground.shape), dtype=np.uint8)
    target = _resolve_patch_size(dummy, patch_size, "patch sampling")
    if target is None:
        return []
    target = tuple(int(i) for i in target)
    spatial_shape = tuple(int(i) for i in foreground.shape)
    if len(spatial_shape) != len(target):
        return []
    max_starts_per_axis = [max(dim - size, 0) for dim, size in zip(spatial_shape, target)]
    strides = [max(1, size // 4) for size in target]
    axes = [list(range(0, max_start + 1, stride)) for max_start, stride in zip(max_starts_per_axis, strides)]
    for axis, max_start in enumerate(max_starts_per_axis):
        if axes[axis][-1] != max_start:
            axes[axis].append(max_start)
    integral = foreground.astype(np.int32)
    for axis in range(integral.ndim):
        integral = integral.cumsum(axis=axis)
    effective_patch_voxels = float(np.prod([min(dim, size) for dim, size in zip(spatial_shape, target)]))
    min_count = effective_patch_voxels * float(min_fraction)
    valid_starts: list[list[int]] = []
    for starts_idx in np.ndindex(*(len(values) for values in axes)):
        starts = tuple(axes[axis][idx] for axis, idx in enumerate(starts_idx))
        if _integral_sum_nd(integral, starts, target) >= min_count:
            valid_starts.append([int(i) for i in starts])
            if len(valid_starts) >= int(max_starts):
                break
    return valid_starts


def _build_patch_sampling_metadata(
    case: TaskPreprocessedCase,
    *,
    patch_sizes: Optional[Dict[str, Sequence[int]]],
    threshold: Optional[float],
    min_fraction: float,
    source: str,
    max_starts: int,
) -> Optional[dict]:
    if patch_sizes is None or len(patch_sizes) == 0 or min_fraction <= 0:
        return None
    sampling_mask = case.patch_sampling_mask
    metadata_mode = "mask" if sampling_mask is not None else "threshold"
    sampling_source = "preprocessing_mask"
    sampling_array = None
    if sampling_mask is None:
        if threshold is None:
            return None
        sampling_source = str(source).lower()
        if sampling_source not in {"image", "target"}:
            _fail_validation("patch sampling source must be either 'image' or 'target'")
        sampling_array = case.patch_sampling_image if sampling_source == "image" else case.patch_sampling_target
        if sampling_array is None:
            sampling_array = case.image if sampling_source == "image" else case.target
        if sampling_array is None:
            return None
    entries: Dict[str, dict] = {}
    for name, patch_size in patch_sizes.items():
        if sampling_mask is not None:
            starts = _compute_patch_sampling_starts_from_mask(
                np.asarray(sampling_mask),
                patch_size,
                min_fraction=float(min_fraction),
                max_starts=int(max_starts),
            )
        else:
            starts = _compute_patch_sampling_starts(
                np.asarray(sampling_array),
                patch_size,
                threshold=float(threshold),
                min_fraction=float(min_fraction),
                max_starts=int(max_starts),
            )
        if starts:
            entries[str(name)] = {
                "patch_size": [int(i) for i in patch_size],
                "starts": starts,
            }
    if not entries:
        return None
    return {
        "mode": metadata_mode,
        "source": sampling_source,
        "threshold": None if threshold is None else float(threshold),
        "min_fraction": float(min_fraction),
        "max_starts": int(max_starts),
        "entries": entries,
    }


def _sample_starts_from_precomputed(case: dict, patch_size: Sequence[int], rng: np.random.RandomState) -> Optional[Tuple[int, ...]]:
    patch_sampling = case.get("patch_sampling")
    if not isinstance(patch_sampling, dict):
        return None
    entries = patch_sampling.get("entries")
    if not isinstance(entries, dict):
        return None
    entry = entries.get(_patch_key(patch_size))
    if not isinstance(entry, dict):
        return None
    starts = entry.get("starts")
    if not isinstance(starts, list) or len(starts) == 0:
        return None
    picked = starts[int(rng.randint(0, len(starts)))]
    return tuple(int(i) for i in picked)


def _normalize_location_samples(locations: Any) -> list[tuple[int, ...]]:
    if locations is None:
        return []
    if isinstance(locations, np.ndarray):
        iterable = locations.tolist()
    elif isinstance(locations, list):
        iterable = locations
    else:
        return []
    normalized: list[tuple[int, ...]] = []
    for row in iterable:
        if isinstance(row, np.ndarray):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            continue
        normalized.append(tuple(int(i) for i in row))
    return normalized


def _gather_sampling_locations(case: dict) -> list[tuple[int, ...]]:
    properties = case.get("properties")
    if isinstance(properties, dict):
        class_locations = properties.get("class_locations")
        if isinstance(class_locations, dict):
            pooled: list[tuple[int, ...]] = []
            for values in class_locations.values():
                pooled.extend(_normalize_location_samples(values))
            if pooled:
                return pooled
    return _normalize_location_samples(case.get("mask_locations"))


def _sample_starts_from_locations(
    case: dict,
    patch_size: Sequence[int],
    rng: np.random.RandomState,
) -> Optional[Tuple[int, ...]]:
    locations = _gather_sampling_locations(case)
    if len(locations) == 0:
        return None
    target = _resolve_patch_size_from_spatial_shape(case["image"].shape[1:], patch_size, "location sampling")
    if target is None:
        return None
    spatial_shape = tuple(int(i) for i in case["image"].shape[1:])
    picked = locations[int(rng.randint(0, len(locations)))]
    spatial_location = picked[-len(target):]
    starts: list[int] = []
    for axis, (shape_dim, patch_dim, center) in enumerate(zip(spatial_shape, target, spatial_location)):
        shape_dim = int(shape_dim)
        patch_dim = int(patch_dim)
        center = int(center)
        if shape_dim <= patch_dim:
            starts.append(0)
            continue
        lower_bound = max(0, center - patch_dim + 1)
        upper_bound = min(center, shape_dim - patch_dim)
        if upper_bound < lower_bound:
            starts.append(max(0, min(shape_dim - patch_dim, center - patch_dim // 2)))
            continue
        starts.append(int(rng.randint(lower_bound, upper_bound + 1)))
    return tuple(starts)


def _foreground_fraction_in_patch(
    array: np.ndarray,
    patch_size: Sequence[int],
    starts: Sequence[int],
    threshold: float,
) -> float:
    patch = np.asarray(_crop_with_starts(array, patch_size, starts))
    if patch.ndim <= 1:
        return float(np.mean(patch > threshold))
    if patch.ndim >= 3:
        foreground = np.any(patch > threshold, axis=0)
    else:
        foreground = patch > threshold
    return float(np.mean(foreground))


def _compute_crop_starts_with_threshold(
    array: np.ndarray,
    patch_size: Sequence[int],
    rng: np.random.RandomState,
    *,
    threshold: Optional[float],
    min_fraction: float,
    max_tries: int,
) -> Tuple[int, ...]:
    starts = _compute_crop_starts(array.shape[1:], patch_size, rng)
    if threshold is None or min_fraction <= 0:
        return starts
    best_starts = starts
    best_fraction = -1.0
    for _ in range(max(1, int(max_tries))):
        starts = _compute_crop_starts(array.shape[1:], patch_size, rng)
        fraction = _foreground_fraction_in_patch(array, patch_size, starts, float(threshold))
        if fraction >= min_fraction:
            return starts
        if fraction > best_fraction:
            best_fraction = fraction
            best_starts = starts
    return best_starts


class TaskPreprocessedDataset(Dataset):
    def __init__(
        self,
        folder: str,
        extra_folder: Optional[str] = None,
        identifiers: Optional[Sequence[str]] = None,
        patch_size: Optional[Sequence[int]] = None,
        transform: Optional[Callable[[Dict], Dict]] = None,
        require_target: bool = False,
        seed: int = 1234,
        patch_foreground_threshold: Optional[float] = None,
        patch_foreground_min_fraction: float = 0.0,
        patch_foreground_source: str = "image",
        patch_foreground_max_tries: int = 32,
    ):
        _require_torch()
        self.folder = folder
        self.extra_folder = extra_folder
        self.identifiers = list(identifiers) if identifiers is not None else _list_identifiers(folder)
        if len(self.identifiers) == 0:
            _fail_validation(f"TaskPreprocessedDataset requires at least one case in folder {folder}")
        self.patch_size = tuple(int(i) for i in patch_size) if patch_size is not None else None
        if transform is not None and not callable(transform):
            _fail_validation("transform must be callable")
        self.transform = transform
        self.require_target = require_target
        self.seed = int(seed)
        self.patch_foreground_threshold = (
            None if patch_foreground_threshold is None else float(patch_foreground_threshold)
        )
        self.patch_foreground_min_fraction = float(patch_foreground_min_fraction)
        self.patch_foreground_source = str(patch_foreground_source).lower()
        self.patch_foreground_max_tries = int(patch_foreground_max_tries)
        if self.patch_foreground_source not in {"image", "target"}:
            _fail_validation("patch_foreground_source must be either 'image' or 'target'")
        if not (0.0 <= self.patch_foreground_min_fraction <= 1.0):
            _fail_validation(
                f"patch_foreground_min_fraction must be in [0, 1], got {self.patch_foreground_min_fraction}"
            )
        if self.patch_foreground_max_tries <= 0:
            _fail_validation(f"patch_foreground_max_tries must be positive, got {self.patch_foreground_max_tries}")

    def __len__(self) -> int:
        return len(self.identifiers)

    def _compute_patch_starts_for_case(
        self,
        case: dict,
        identifier: str,
        rng: np.random.RandomState,
    ) -> Optional[Tuple[int, ...]]:
        if self.patch_size is None:
            return None
        if case["target"] is not None and tuple(case["image"].shape[1:]) != tuple(case["target"].shape[1:]):
            _fail_validation(
                f"Case '{identifier}' has mismatched image/target shapes: "
                f"{case['image'].shape[1:]} vs {case['target'].shape[1:]}"
            )
        crop_reference = case["image"]
        if self.patch_foreground_source == "target":
            if case["target"] is None:
                _fail_validation(
                    "patch_foreground_source='target' requires targets to be present for patch sampling"
                )
            crop_reference = case["target"]
        starts = _sample_starts_from_locations(case, self.patch_size, rng)
        if starts is None:
            starts = _sample_starts_from_precomputed(case, self.patch_size, rng)
        if starts is None:
            starts = _compute_crop_starts_with_threshold(
                crop_reference,
                self.patch_size,
                rng,
                threshold=self.patch_foreground_threshold,
                min_fraction=self.patch_foreground_min_fraction,
                max_tries=self.patch_foreground_max_tries,
            )
        return starts

    def get_target_only(self, index: int) -> Dict[str, Any]:
        identifier = self.identifiers[index]
        case = load_preprocessed_case(self.folder, identifier, extra_folder=self.extra_folder)
        if case["target"] is None:
            _fail_validation(f"Case '{identifier}' does not have a target")
        rng = _build_runtime_rng(self.seed, index)
        starts = self._compute_patch_starts_for_case(case, identifier, rng)
        target = case["target"] if starts is None else _crop_with_starts(case["target"], self.patch_size, starts)
        target_tensor = torch.from_numpy(np.asarray(target))
        if np.issubdtype(target.dtype, np.integer):
            target_tensor = target_tensor.long()
        else:
            target_tensor = target_tensor.float()
        return {
            "target": target_tensor,
            "identifier": identifier,
        }

    def __getitem__(self, index: int) -> Dict:
        identifier = self.identifiers[index]
        case = load_preprocessed_case(self.folder, identifier, extra_folder=self.extra_folder)
        if self.require_target and case["target"] is None:
            _fail_validation(f"Case '{identifier}' does not have a target")
        rng = _build_runtime_rng(self.seed, index)
        if self.patch_size is None:
            image = case["image"]
            target = case["target"]
            starts = None
        else:
            starts = self._compute_patch_starts_for_case(case, identifier, rng)
            image = _crop_with_starts(case["image"], self.patch_size, starts)
            target = _crop_with_starts(case["target"], self.patch_size, starts) if case["target"] is not None else None
        evaluation_reference = case.get("evaluation_reference")
        if evaluation_reference is not None:
            if tuple(case["image"].shape[1:]) != tuple(evaluation_reference.shape[1:]):
                _fail_validation(
                    f"Case '{identifier}' has mismatched image/evaluation_reference shapes: "
                    f"{case['image'].shape[1:]} vs {evaluation_reference.shape[1:]}"
                )
            if self.patch_size is not None:
                evaluation_reference = _crop_with_starts(evaluation_reference, self.patch_size, starts)
        stored_mask = case.get("mask")
        if stored_mask is not None and self.patch_size is not None:
            stored_mask = _crop_with_starts(stored_mask[None], self.patch_size, starts)[0]
        conflict_map = case.get("conflict_map")
        if conflict_map is not None and self.patch_size is not None:
            conflict_map = _crop_with_starts(conflict_map, self.patch_size, starts)
        sample = {
            "image": torch.from_numpy(np.asarray(image)).float(),
            "identifier": identifier,
            "properties": case["properties"],
            "task_mode": case.get("task_mode"),
            "run_stage": case.get("run_stage"),
            "reference_type": case.get("reference_type"),
        }
        if target is not None:
            sample["target"] = torch.from_numpy(np.asarray(target))
            if np.issubdtype(target.dtype, np.integer):
                sample["target"] = sample["target"].long()
            else:
                sample["target"] = sample["target"].float()
        if evaluation_reference is not None:
            sample["evaluation_reference"] = torch.from_numpy(np.asarray(evaluation_reference))
            if np.issubdtype(evaluation_reference.dtype, np.integer):
                sample["evaluation_reference"] = sample["evaluation_reference"].long()
            else:
                sample["evaluation_reference"] = sample["evaluation_reference"].float()
            sample["evaluation_properties"] = case.get("evaluation_properties")
        if stored_mask is not None:
            sample["mask"] = torch.from_numpy(np.asarray(stored_mask != 0)).bool()
        if conflict_map is not None:
            sample["conflict_map"] = torch.from_numpy(np.asarray(conflict_map)).float()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class SegmentationDataset(TaskPreprocessedDataset):
    def __init__(self, *args, **kwargs):
        kwargs["require_target"] = True
        super().__init__(*args, **kwargs)


class PairedGenerativeDataset(TaskPreprocessedDataset):
    def __init__(self, *args, **kwargs):
        kwargs["require_target"] = True
        super().__init__(*args, **kwargs)


class SelfSupervisedDataset(TaskPreprocessedDataset):
    def __init__(self, *args, view_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.view_transform = view_transform

    def __getitem__(self, index: int) -> Dict:
        sample = super().__getitem__(index)
        image = sample["image"]
        if self.view_transform is None:
            sample["view1"] = image.clone()
            sample["view2"] = image.clone()
        else:
            sample["view1"] = self.view_transform(image.clone())
            sample["view2"] = self.view_transform(image.clone())
        return sample


class UnpairedGenerativeDataset(Dataset):
    def __init__(
        self,
        folder_a: str,
        folder_b: str,
        identifiers_a: Optional[Sequence[str]] = None,
        identifiers_b: Optional[Sequence[str]] = None,
        patch_size: Optional[Sequence[int]] = None,
        transform: Optional[Callable[[Dict], Dict]] = None,
        random_pairing: bool = True,
        seed: int = 1234,
    ):
        _require_torch()
        self.folder_a = folder_a
        self.folder_b = folder_b
        self.identifiers_a = list(identifiers_a) if identifiers_a is not None else _list_identifiers(folder_a)
        self.identifiers_b = list(identifiers_b) if identifiers_b is not None else _list_identifiers(folder_b)
        if len(self.identifiers_a) == 0:
            _fail_validation("UnpairedGenerativeDataset requires at least one preprocessed case in domain A")
        if len(self.identifiers_b) == 0:
            _fail_validation("UnpairedGenerativeDataset requires at least one preprocessed case in domain B")
        self.patch_size = tuple(int(i) for i in patch_size) if patch_size is not None else None
        if transform is not None and not callable(transform):
            _fail_validation("transform must be callable")
        self.transform = transform
        self.random_pairing = random_pairing
        self.seed = int(seed)

    def __len__(self) -> int:
        return max(len(self.identifiers_a), len(self.identifiers_b))

    def __getitem__(self, index: int) -> Dict:
        rng = _build_runtime_rng(self.seed, index)
        identifier_a = self.identifiers_a[index % len(self.identifiers_a)]
        identifier_b = (
            self.identifiers_b[int(rng.randint(0, len(self.identifiers_b)))]
            if self.random_pairing
            else self.identifiers_b[index % len(self.identifiers_b)]
        )
        case_a = load_preprocessed_case(self.folder_a, identifier_a)
        case_b = load_preprocessed_case(self.folder_b, identifier_b)
        starts_a = _sample_starts_from_locations(case_a, self.patch_size, rng) if self.patch_size is not None else None
        if starts_a is None and self.patch_size is not None:
            starts_a = _sample_starts_from_precomputed(case_a, self.patch_size, rng)
        starts_b = _sample_starts_from_locations(case_b, self.patch_size, rng) if self.patch_size is not None else None
        if starts_b is None and self.patch_size is not None:
            starts_b = _sample_starts_from_precomputed(case_b, self.patch_size, rng)
        sample = {
            "image_a": torch.from_numpy(
                np.asarray(
                    _crop_with_starts(case_a["image"], self.patch_size, starts_a)
                    if starts_a is not None
                    else _crop_or_pad(case_a["image"], self.patch_size, rng)
                )
            ).float(),
            "image_b": torch.from_numpy(
                np.asarray(
                    _crop_with_starts(case_b["image"], self.patch_size, starts_b)
                    if starts_b is not None
                    else _crop_or_pad(case_b["image"], self.patch_size, rng)
                )
            ).float(),
            "identifier_a": identifier_a,
            "identifier_b": identifier_b,
            "properties_a": case_a["properties"],
            "properties_b": case_b["properties"],
        }
        if case_a.get("mask") is not None:
            mask_a = (
                _crop_with_starts(case_a["mask"][None], self.patch_size, starts_a)[0]
                if starts_a is not None
                else _crop_or_pad(case_a["mask"][None], self.patch_size, rng)[0]
            )
            sample["mask_a"] = torch.from_numpy(np.asarray(mask_a != 0)).bool()
        if case_b.get("mask") is not None:
            mask_b = (
                _crop_with_starts(case_b["mask"][None], self.patch_size, starts_b)[0]
                if starts_b is not None
                else _crop_or_pad(case_b["mask"][None], self.patch_size, rng)[0]
            )
            sample["mask_b"] = torch.from_numpy(np.asarray(mask_b != 0)).bool()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def load_preprocessed_dataset(
    folder: str,
    *,
    extra_folder: Optional[str] = None,
    patch_size: Optional[Sequence[int]] = None,
    use_manifest_patch_size: bool = True,
    configuration: Optional[str] = None,
    split: Optional[str] = None,
    transform: Optional[Callable[[Dict], Dict]] = None,
    seed: int = 1234,
    random_pairing: Optional[bool] = None,
    view_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    patch_foreground_threshold: Optional[float] = None,
    patch_foreground_min_fraction: float = 0.0,
    patch_foreground_source: str = "image",
    patch_foreground_max_tries: int = 32,
) -> Dataset:
    manifest = load_preprocessed_dataset_manifest(folder)
    if split is not None and split not in {"train", "val"}:
        _fail_validation(f"split must be one of None, 'train', or 'val', got '{split}'")
    manifest_patch_size = manifest.get("default_patch_size")
    manifest_configurations = manifest.get("configurations") or {}
    effective_configuration = configuration if configuration is not None else manifest.get("default_configuration")
    if effective_configuration is not None and effective_configuration not in manifest_configurations:
        _fail_validation(f"Requested configuration '{effective_configuration}' is not present in the manifest")
    effective_patch_size = patch_size
    if use_manifest_patch_size and effective_patch_size is None and effective_configuration is not None:
        effective_patch_size = manifest_configurations.get(effective_configuration, {}).get("patch_size")
    if use_manifest_patch_size and effective_patch_size is None:
        effective_patch_size = manifest_patch_size
    task_mode = manifest["task_mode"]
    run_stage = manifest["run_stage"]

    if manifest["dataset_kind"] == "unpaired_domains":
        domains = manifest["domains"]
        folder_a = os.path.join(folder, domains["a"]["folder"])
        folder_b = os.path.join(folder, domains["b"]["folder"])
        identifiers_a = domains["a"].get("identifiers")
        identifiers_b = domains["b"].get("identifiers")
        if split is not None:
            splits_a = domains["a"].get("splits")
            splits_b = domains["b"].get("splits")
            if splits_a is None or splits_b is None:
                _fail_validation("Requested split loading but the manifest does not contain unpaired domain splits")
            identifiers_a = splits_a[split]
            identifiers_b = splits_b[split]
        return UnpairedGenerativeDataset(
            folder_a=folder_a,
            folder_b=folder_b,
            identifiers_a=identifiers_a,
            identifiers_b=identifiers_b,
            patch_size=effective_patch_size,
            transform=transform,
            random_pairing=manifest.get("random_pairing", True) if random_pairing is None else random_pairing,
            seed=seed,
        )

    identifiers = manifest.get("identifiers")
    if split is not None:
        splits = manifest.get("splits")
        if splits is None:
            _fail_validation("Requested split loading but the manifest does not contain train/val splits")
        identifiers = splits[split]
    common_kwargs: Dict[str, Any] = {
        "folder": folder,
        "extra_folder": extra_folder,
        "identifiers": identifiers,
        "patch_size": effective_patch_size,
        "transform": transform,
        "seed": seed,
        "patch_foreground_threshold": patch_foreground_threshold,
        "patch_foreground_min_fraction": patch_foreground_min_fraction,
        "patch_foreground_source": patch_foreground_source,
        "patch_foreground_max_tries": patch_foreground_max_tries,
    }
    if task_mode == TaskMode.SEGMENTATION:
        return SegmentationDataset(**common_kwargs) if run_stage == RunStage.TRAIN else TaskPreprocessedDataset(**common_kwargs)
    if task_mode == TaskMode.PAIRED_GENERATIVE:
        return (
            PairedGenerativeDataset(**common_kwargs)
            if run_stage == RunStage.TRAIN
            else TaskPreprocessedDataset(**common_kwargs)
        )
    if task_mode == TaskMode.SELF_SUPERVISED:
        return SelfSupervisedDataset(**common_kwargs, view_transform=view_transform)
    _fail_validation(f"Unsupported task_mode '{task_mode}' in manifest")
