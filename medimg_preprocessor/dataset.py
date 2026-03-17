from __future__ import annotations

from dataclasses import asdict
import json
import os
import pickle
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:
        pass

from .config import PreprocessingConfig
from .preprocessing import RunStage, TaskMode, TaskPreprocessedCase


MANIFEST_FILENAME = "preprocessing_manifest.json"


def _fail_validation(message: str) -> None:
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)


def _require_torch() -> None:
    if torch is None:
        _fail_validation("torch is required for dataset loading and tensor conversion")


def _serialize_config(config: Optional[PreprocessingConfig]) -> Optional[dict]:
    if config is None:
        return None
    if not isinstance(config, PreprocessingConfig):
        _fail_validation(f"config must be a PreprocessingConfig, got {type(config).__name__}")
    return asdict(config)


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


def save_preprocessed_case(case: TaskPreprocessedCase, output_filename_truncated: str) -> None:
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
    arrays = {"image": np.ascontiguousarray(case.image)}
    if case.target is not None:
        arrays["target"] = np.ascontiguousarray(case.target)
    if case.evaluation_reference is not None:
        arrays["evaluation_reference"] = np.ascontiguousarray(case.evaluation_reference)
    np.savez_compressed(output_filename_truncated + ".npz", **arrays)
    metadata = {
        "properties": case.properties,
        "target_properties": case.target_properties,
        "evaluation_properties": case.evaluation_properties,
        "task_mode": case.task_mode,
        "run_stage": case.run_stage,
        "reference_type": case.reference_type,
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
    identifiers: Optional[Sequence[str]] = None,
) -> str:
    if not os.path.isdir(folder):
        _fail_validation(f"Preprocessed dataset folder does not exist: {folder}")
    _validate_task_mode(task_mode)
    _validate_run_stage(run_stage)
    if task_mode == TaskMode.UNPAIRED_GENERATIVE:
        _fail_validation("Use save_unpaired_preprocessed_dataset_manifest for unpaired_generative datasets")
    manifest = {
        "format_version": 1,
        "task_mode": task_mode,
        "run_stage": run_stage,
        "dataset_kind": "single_folder",
        "default_patch_size": None if default_patch_size is None else [int(i) for i in default_patch_size],
        "identifiers": list(identifiers) if identifiers is not None else _list_identifiers(folder),
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
    identifiers: Optional[Sequence[str]] = None,
    folder_a: Optional[str] = None,
    folder_b: Optional[str] = None,
    config_a: Optional[PreprocessingConfig] = None,
    config_b: Optional[PreprocessingConfig] = None,
    identifiers_a: Optional[Sequence[str]] = None,
    identifiers_b: Optional[Sequence[str]] = None,
    random_pairing: bool = True,
) -> str:
    _validate_task_mode(task_mode)
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
            identifiers_a=identifiers_a,
            identifiers_b=identifiers_b,
            random_pairing=random_pairing,
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
        identifiers=identifiers,
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
    identifiers_a: Optional[Sequence[str]] = None,
    identifiers_b: Optional[Sequence[str]] = None,
    random_pairing: bool = True,
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
    manifest = {
        "format_version": 1,
        "task_mode": TaskMode.UNPAIRED_GENERATIVE,
        "run_stage": run_stage,
        "dataset_kind": "unpaired_domains",
        "default_patch_size": None if default_patch_size is None else [int(i) for i in default_patch_size],
        "random_pairing": bool(random_pairing),
        "domains": {
            "a": {
                "folder": os.path.relpath(folder_a_abs, root_folder),
                "identifiers": list(identifiers_a) if identifiers_a is not None else _list_identifiers(folder_a_abs),
                "preprocessing_config": _serialize_config(config_a),
            },
            "b": {
                "folder": os.path.relpath(folder_b_abs, root_folder),
                "identifiers": list(identifiers_b) if identifiers_b is not None else _list_identifiers(folder_b_abs),
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
    if task_mode == TaskMode.UNPAIRED_GENERATIVE and dataset_kind != "unpaired_domains":
        _fail_validation("unpaired_generative manifest must use dataset_kind 'unpaired_domains'")
    if task_mode != TaskMode.UNPAIRED_GENERATIVE and dataset_kind != "single_folder":
        _fail_validation(f"{task_mode} manifest must use dataset_kind 'single_folder'")
    if manifest.get("default_patch_size") is not None and not isinstance(manifest["default_patch_size"], list):
        _fail_validation("Manifest 'default_patch_size' must be a list when provided")
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
    return manifest


def load_preprocessed_case(folder: str, identifier: str) -> dict:
    if not os.path.isdir(folder):
        _fail_validation(f"Preprocessed case folder does not exist: {folder}")
    npz_file = os.path.join(folder, identifier + ".npz")
    pkl_file = os.path.join(folder, identifier + ".pkl")
    if not os.path.isfile(npz_file):
        _fail_validation(f"Preprocessed array file does not exist: {npz_file}")
    if not os.path.isfile(pkl_file):
        _fail_validation(f"Preprocessed metadata file does not exist: {pkl_file}")
    with np.load(npz_file) as arrays:
        with open(pkl_file, "rb") as f:
            meta = pickle.load(f)
        if "image" not in arrays.files:
            _fail_validation(f"Preprocessed case '{identifier}' does not contain an 'image' array")
        if not isinstance(meta, dict):
            _fail_validation(f"Metadata for case '{identifier}' must be a dict")
        case = {
            "image": arrays["image"],
            "target": arrays["target"] if "target" in arrays.files else None,
            "evaluation_reference": arrays["evaluation_reference"]
            if "evaluation_reference" in arrays.files
            else None,
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
        return case


def _list_identifiers(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        _fail_validation(f"Preprocessed dataset folder does not exist: {folder}")
    return sorted([os.path.splitext(i)[0] for i in os.listdir(folder) if i.endswith(".npz")])


def _validate_patch_dims(array: np.ndarray, patch_size: Optional[Sequence[int]], context: str) -> None:
    if patch_size is None:
        return
    spatial_dims = array.ndim - 1
    if len(patch_size) != spatial_dims:
        _fail_validation(
            f"{context} expects patch_size with {spatial_dims} spatial dims, got {len(patch_size)} "
            f"for array shape {array.shape}"
        )


def _crop_or_pad(array: np.ndarray, patch_size: Optional[Sequence[int]], rng: np.random.RandomState) -> np.ndarray:
    if patch_size is None:
        return array
    _validate_patch_dims(array, patch_size, "crop/pad")
    target = tuple(int(i) for i in patch_size)
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
    slicer = (slice(None),) + tuple(slice(s, s + p) for s, p in zip(starts, target))
    return array[slicer]


def _pad_to_patch_size(array: np.ndarray, patch_size: Sequence[int]) -> np.ndarray:
    _validate_patch_dims(array, patch_size, "crop/pad")
    target = tuple(int(i) for i in patch_size)
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
    if len(spatial_shape) != len(patch_size):
        _fail_validation(
            f"crop/pad expects patch_size with {len(spatial_shape)} spatial dims, got {len(patch_size)} "
            f"for spatial shape {tuple(spatial_shape)}"
        )
    starts = []
    for current, wanted in zip(spatial_shape, patch_size):
        current = max(current, wanted)
        starts.append(0 if current == wanted else int(rng.randint(0, current - wanted + 1)))
    return tuple(starts)


def _crop_with_starts(array: np.ndarray, patch_size: Sequence[int], starts: Sequence[int]) -> np.ndarray:
    _validate_patch_dims(array, patch_size, "crop/pad")
    target = tuple(int(i) for i in patch_size)
    if len(starts) != len(target):
        _fail_validation(f"crop/pad received {len(starts)} crop starts for patch_size with {len(target)} dims")
    array = _pad_to_patch_size(array, target)
    slicer = (slice(None),) + tuple(slice(s, s + p) for s, p in zip(starts, target))
    return array[slicer]


class TaskPreprocessedDataset(Dataset):
    def __init__(
        self,
        folder: str,
        identifiers: Optional[Sequence[str]] = None,
        patch_size: Optional[Sequence[int]] = None,
        transform: Optional[Callable[[Dict], Dict]] = None,
        require_target: bool = False,
        seed: int = 1234,
    ):
        _require_torch()
        self.folder = folder
        self.identifiers = list(identifiers) if identifiers is not None else _list_identifiers(folder)
        if len(self.identifiers) == 0:
            _fail_validation(f"TaskPreprocessedDataset requires at least one case in folder {folder}")
        self.patch_size = tuple(int(i) for i in patch_size) if patch_size is not None else None
        if transform is not None and not callable(transform):
            _fail_validation("transform must be callable")
        self.transform = transform
        self.require_target = require_target
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int) -> Dict:
        identifier = self.identifiers[index]
        case = load_preprocessed_case(self.folder, identifier)
        if self.require_target and case["target"] is None:
            _fail_validation(f"Case '{identifier}' does not have a target")
        rng = np.random.RandomState(self.seed + index)
        if self.patch_size is None:
            image = case["image"]
            target = case["target"]
        else:
            if case["target"] is not None and tuple(case["image"].shape[1:]) != tuple(case["target"].shape[1:]):
                _fail_validation(
                    f"Case '{identifier}' has mismatched image/target shapes: "
                    f"{case['image'].shape[1:]} vs {case['target'].shape[1:]}"
                )
            starts = _compute_crop_starts(case["image"].shape[1:], self.patch_size, rng)
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
        rng = np.random.RandomState(self.seed + index)
        identifier_a = self.identifiers_a[index % len(self.identifiers_a)]
        identifier_b = (
            self.identifiers_b[int(rng.randint(0, len(self.identifiers_b)))]
            if self.random_pairing
            else self.identifiers_b[index % len(self.identifiers_b)]
        )
        case_a = load_preprocessed_case(self.folder_a, identifier_a)
        case_b = load_preprocessed_case(self.folder_b, identifier_b)
        sample = {
            "image_a": torch.from_numpy(np.asarray(_crop_or_pad(case_a["image"], self.patch_size, rng))).float(),
            "image_b": torch.from_numpy(np.asarray(_crop_or_pad(case_b["image"], self.patch_size, rng))).float(),
            "identifier_a": identifier_a,
            "identifier_b": identifier_b,
            "properties_a": case_a["properties"],
            "properties_b": case_b["properties"],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def load_preprocessed_dataset(
    folder: str,
    *,
    patch_size: Optional[Sequence[int]] = None,
    transform: Optional[Callable[[Dict], Dict]] = None,
    seed: int = 1234,
    random_pairing: Optional[bool] = None,
    view_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dataset:
    manifest = load_preprocessed_dataset_manifest(folder)
    manifest_patch_size = manifest.get("default_patch_size")
    effective_patch_size = patch_size if patch_size is not None else manifest_patch_size
    task_mode = manifest["task_mode"]
    run_stage = manifest["run_stage"]

    if manifest["dataset_kind"] == "unpaired_domains":
        domains = manifest["domains"]
        folder_a = os.path.join(folder, domains["a"]["folder"])
        folder_b = os.path.join(folder, domains["b"]["folder"])
        return UnpairedGenerativeDataset(
            folder_a=folder_a,
            folder_b=folder_b,
            identifiers_a=domains["a"].get("identifiers"),
            identifiers_b=domains["b"].get("identifiers"),
            patch_size=effective_patch_size,
            transform=transform,
            random_pairing=manifest.get("random_pairing", True) if random_pairing is None else random_pairing,
            seed=seed,
        )

    identifiers = manifest.get("identifiers")
    common_kwargs: Dict[str, Any] = {
        "folder": folder,
        "identifiers": identifiers,
        "patch_size": effective_patch_size,
        "transform": transform,
        "seed": seed,
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
