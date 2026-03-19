from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
from time import sleep
from pathlib import Path
from typing import Optional, Sequence

from .config import PreprocessingConfig, ResamplingConfig
from .dataset import (
    load_preprocessed_dataset_manifest,
    save_preprocessed_case,
    save_preprocessed_dataset,
)
from .planning import plan_preprocessing_from_cases
from .preprocessing import RunStage, TaskMode


DESCRIPTION = """medimg_preprocessor CLI

주요 명령:

1. preprocess-dataset
   raw 디렉토리를 스캔하고, 자동 planning 후 전처리 결과와
   preprocessing_manifest.json을 생성합니다.
   config나 plans를 따로 주지 않으면 nnU-Net 스타일로 자동 결정합니다.

2. save-dataset
   이미 저장된 전처리 case 파일들에 대해
   preprocessing_manifest.json만 다시 생성합니다.

3. show-manifest
   저장된 preprocessing_manifest.json 내용을 출력합니다.
"""


PREPROCESS_DATASET_EPILOG = """예시:
  Segmentation:
    python -m medimg_preprocessor preprocess-dataset --task-mode segmentation --images-dir raw/imagesTr --labels-dir raw/labelsTr --output-folder preprocessed_seg

  Segmentation with multi-image inputs:
    python -m medimg_preprocessor preprocess-dataset --task-mode segmentation --images-dir raw/imagesTr --labels-dir raw/labelsTr --output-folder preprocessed_seg --multi-image

  Paired generative:
    python -m medimg_preprocessor preprocess-dataset --task-mode paired_generative --source-dir raw/source --target-dir raw/target --output-folder preprocessed_paired

  Unpaired generative:
    python -m medimg_preprocessor preprocess-dataset --task-mode unpaired_generative --domain-a-dir raw/domain_a --domain-b-dir raw/domain_b --output-folder preprocessed_unpaired
"""


SAVE_DATASET_EPILOG = """예시:
  Single-folder segmentation dataset:
    python -m medimg_preprocessor save-dataset --folder preprocessed_seg --task-mode segmentation --run-stage train --plans-file nnUNetPlans.json --configuration-name 3d_fullres --default-patch-size 96 96 96

  Single-folder paired generative dataset:
    python -m medimg_preprocessor save-dataset --folder preprocessed_paired --task-mode paired_generative --run-stage train --config-json config.json

  Unpaired generative dataset:
    python -m medimg_preprocessor save-dataset --folder preprocessed_unpaired --task-mode unpaired_generative --folder-a domain_a --folder-b domain_b --config-a-json config_a.json --config-b-json config_b.json
"""


SHOW_MANIFEST_EPILOG = """예시:
  python -m medimg_preprocessor show-manifest --folder preprocessed_seg
"""


READER_CHOICES = (
    "auto",
    "nibabel",
    "nibabel_reorient",
    "simpleitk",
    "simpleitk_reorient",
    "tiff3d",
    "natural_2d",
)

SUPPORTED_SCAN_ENDINGS = (
    ".nii.gz",
    ".nii",
    ".nrrd",
    ".mha",
    ".gipl",
    ".tiff",
    ".tif",
    ".png",
    ".bmp",
)

MULTI_IMAGE_PATTERN = re.compile(r"^(?P<identifier>.+_\d{4})_(?P<channel>\d{4})$")


def _log_stage(step: int, total_steps: int, title: str, detail: Optional[str] = None) -> None:
    message = f"[{step}/{total_steps}] {title}"
    if detail:
        message += f": {detail}"
    print(message, flush=True)


def _format_progress(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        total = 1
    filled = int(round(width * current / total))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total}"


def _run_case_progress(
    label: str,
    work_items: Sequence[dict],
    worker_fn,
    num_processes: int,
) -> None:
    total = len(work_items)
    if total == 0:
        return
    if num_processes <= 1:
        for index, item in enumerate(work_items, start=1):
            sys.stdout.write(f"\r  {label:<12} {_format_progress(index, total)}")
            sys.stdout.flush()
            worker_fn(item)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = [pool.apply_async(worker_fn, (item,)) for item in work_items]
        remaining = list(range(total))
        workers = [j for j in pool._pool]
        while len(remaining) > 0:
            all_alive = all([j.is_alive() for j in workers])
            if not all_alive:
                raise RuntimeError(
                    "A background preprocessing worker died unexpectedly. "
                    "This is often caused by an exception or by running out of memory."
                )
            done = [i for i in remaining if results[i].ready()]
            for i in done:
                results[i].get()
            completed = total - len(remaining) + len(done)
            sys.stdout.write(f"\r  {label:<12} {_format_progress(completed, total)}")
            sys.stdout.flush()
            remaining = [i for i in remaining if i not in done]
            sleep(0.1)
        sys.stdout.write("\n")
        sys.stdout.flush()


def _ensure_storage_runtime(storage_format: str) -> None:
    if storage_format == "blosc2":
        try:
            import blosc2  # noqa: F401
        except ModuleNotFoundError as e:
            raise ValueError(
                "storage_format='blosc2' requires the 'blosc2' package, but it is not installed in the current environment."
            ) from e


def _load_config_from_json(path: Optional[str]) -> Optional[PreprocessingConfig]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config JSON must contain an object, got {type(payload).__name__}")
    resampling_payload = payload.get("resampling", {})
    if not isinstance(resampling_payload, dict):
        raise ValueError("Config JSON field 'resampling' must be an object when provided")
    return PreprocessingConfig(
        spacing=payload["spacing"],
        transpose_forward=payload["transpose_forward"],
        normalization_schemes=payload["normalization_schemes"],
        use_mask_for_norm=payload["use_mask_for_norm"],
        foreground_intensity_properties_per_channel=payload.get(
            "foreground_intensity_properties_per_channel",
            {},
        ),
        resampling=ResamplingConfig(
            image_order=int(resampling_payload.get("image_order", 3)),
            image_order_z=int(resampling_payload.get("image_order_z", 0)),
            label_order=int(resampling_payload.get("label_order", 0)),
            label_order_z=int(resampling_payload.get("label_order_z", 0)),
            force_separate_z=resampling_payload.get("force_separate_z", None),
            separate_z_anisotropy_threshold=float(
                resampling_payload.get("separate_z_anisotropy_threshold", 3.0)
            ),
        ),
    )


def _load_config(
    config_json: Optional[str],
    plans_file: Optional[str],
    configuration_name: Optional[str],
) -> Optional[PreprocessingConfig]:
    if config_json is not None and plans_file is not None:
        raise ValueError("Use either --config-json or --plans-file, not both")
    if config_json is not None:
        return _load_config_from_json(config_json)
    if plans_file is not None:
        if configuration_name is None:
            raise ValueError("--configuration-name is required when using --plans-file")
        return PreprocessingConfig.from_nnunet_plans(plans_file, configuration_name)
    if configuration_name is not None:
        raise ValueError("--configuration-name requires --plans-file")
    return None


def _strip_known_suffix(filename: str) -> str:
    lower = filename.lower()
    for ending in SUPPORTED_SCAN_ENDINGS:
        if lower.endswith(ending):
            return filename[: -len(ending)]
    return Path(filename).stem


def _detect_file_ending(filename: str) -> str:
    lower = filename.lower()
    for ending in SUPPORTED_SCAN_ENDINGS:
        if lower.endswith(ending):
            return ending
    return Path(filename).suffix.lower()


def _build_reader(reader_name: str, example_file: str, dataset_json_content: Optional[dict] = None):
    from .imageio import (
        NaturalImage2DIO,
        NibabelIO,
        NibabelIOWithReorient,
        SimpleITKIO,
        SimpleITKIOWithReorient,
        Tiff3DIO,
        determine_reader_writer_from_dataset_json,
        determine_reader_writer_from_file_ending,
    )

    registry = {
        "nibabel": NibabelIO,
        "nibabel_reorient": NibabelIOWithReorient,
        "simpleitk": SimpleITKIO,
        "simpleitk_reorient": SimpleITKIOWithReorient,
        "tiff3d": Tiff3DIO,
        "natural_2d": NaturalImage2DIO,
    }
    if reader_name == "auto":
        if dataset_json_content is not None:
            return determine_reader_writer_from_dataset_json(
                dataset_json_content,
                example_file=example_file,
                verbose=False,
            )()
        ending = _detect_file_ending(example_file)
        return determine_reader_writer_from_file_ending(ending, example_file=example_file, verbose=False)()
    if reader_name not in registry:
        raise ValueError(f"Unsupported reader '{reader_name}'")
    return registry[reader_name]()


def _prepare_output_prefix(output_folder: Path, identifier: str) -> Path:
    output_prefix = output_folder / identifier
    existing = [
        output_prefix.with_suffix(".npz"),
        output_prefix.with_suffix(".pkl"),
        output_prefix.with_suffix(".b2nd"),
        output_folder / f"{identifier}_target.b2nd",
        output_folder / f"{identifier}_evalref.b2nd",
    ]
    if any(path.exists() for path in existing):
        raise ValueError(f"Refusing to overwrite existing preprocessed case '{identifier}' in {output_folder}")
    return output_prefix


def _load_json_file(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return payload


def _discover_dataset_json(*directories: Optional[str]) -> Optional[dict]:
    candidates: list[Path] = []
    for directory in directories:
        if directory is None:
            continue
        path = Path(directory)
        if path.is_dir():
            candidates.append(path / "dataset.json")
            candidates.append(path.parent / "dataset.json")
    for candidate in candidates:
        payload = _load_json_file(candidate)
        if payload is not None:
            return payload
    return None


def _plan_config_from_cases(
    cases: dict[str, list[str]],
    reader_name: str,
    *,
    dataset_json: Optional[dict] = None,
    reference_cases: Optional[dict[str, str]] = None,
    num_processes: int = 1,
) -> tuple[PreprocessingConfig, dict]:
    first_identifier = sorted(cases.keys())[0]
    reader = _build_reader(reader_name, cases[first_identifier][0], dataset_json)
    return plan_preprocessing_from_cases(
        cases,
        reader,
        dataset_json=dataset_json,
        reference_cases=reference_cases,
        num_processes=num_processes,
    )


def _resolve_default_configuration(configurations: Optional[dict]) -> Optional[str]:
    if not configurations:
        return None
    if "3d" in configurations:
        return "3d"
    if "2d" in configurations:
        return "2d"
    return next(iter(configurations.keys()))


def _build_patch_sampling_patch_sizes(
    default_patch_size: Optional[Sequence[int]],
    configurations: Optional[dict],
) -> Optional[dict[str, tuple[int, ...]]]:
    patch_sizes: dict[str, tuple[int, ...]] = {}
    if default_patch_size is not None:
        patch_sizes["default"] = tuple(int(i) for i in default_patch_size)
    if configurations:
        for name, payload in configurations.items():
            patch_size = payload.get("patch_size")
            if patch_size is not None:
                patch_sizes[str(name)] = tuple(int(i) for i in patch_size)
    return patch_sizes or None


def _merge_unpaired_configurations(configurations_a: Optional[dict], configurations_b: Optional[dict]) -> Optional[dict]:
    if not configurations_a and not configurations_b:
        return None
    if not configurations_a:
        return configurations_b
    if not configurations_b:
        return configurations_a
    shared = {}
    for name in sorted(set(configurations_a.keys()).intersection(configurations_b.keys())):
        patch_a = configurations_a[name].get("patch_size")
        patch_b = configurations_b[name].get("patch_size")
        if patch_a is None or patch_b is None or len(patch_a) != len(patch_b):
            continue
        shared[name] = {
            "patch_size": [min(int(a), int(b)) for a, b in zip(patch_a, patch_b)],
            "spacing": configurations_a[name].get("spacing"),
            "median_shape": configurations_a[name].get("median_shape"),
            "recommended_batch_size": min(
                int(configurations_a[name].get("recommended_batch_size", 2)),
                int(configurations_b[name].get("recommended_batch_size", 2)),
            ),
        }
    return shared or configurations_a


def _list_supported_files(folder: str, label: str) -> list[Path]:
    directory = Path(folder)
    if not directory.is_dir():
        raise ValueError(f"{label} does not exist or is not a directory: {directory}")
    files = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and any(path.name.lower().endswith(ending) for ending in SUPPORTED_SCAN_ENDINGS)
    ]
    if len(files) == 0:
        raise ValueError(f"{label} does not contain any supported image files: {directory}")
    return files


def _scan_single_image_dir(folder: str, label: str) -> dict[str, list[str]]:
    cases: dict[str, list[str]] = {}
    for path in _list_supported_files(folder, label):
        stem = _strip_known_suffix(path.name)
        match = MULTI_IMAGE_PATTERN.match(stem)
        if match is not None:
            identifier = match.group("identifier")
            channel = int(match.group("channel"))
            if channel != 0:
                raise ValueError(
                    f"{label} file '{path.name}' looks like a multi-image channel file. "
                    "Use --multi-image for cases with postfixes beyond _0000."
                )
        else:
            identifier = stem
        if identifier in cases:
            raise ValueError(
                f"{label} contains multiple files for identifier '{identifier}'. "
                "Use --multi-image if these files are channel postfixes like case_0001_0000."
            )
        cases[identifier] = [str(path)]
    return cases


def _scan_multi_image_dir(folder: str, label: str) -> dict[str, list[str]]:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for path in _list_supported_files(folder, label):
        stem = _strip_known_suffix(path.name)
        match = MULTI_IMAGE_PATTERN.match(stem)
        if match is None:
            raise ValueError(
                f"{label} file '{path.name}' does not match the required multi-image postfix rule "
                "(for example case_0001_0000, case_0001_0001)."
            )
        identifier = match.group("identifier")
        channel = int(match.group("channel"))
        grouped.setdefault(identifier, []).append((channel, str(path)))

    cases: dict[str, list[str]] = {}
    for identifier, items in grouped.items():
        items = sorted(items, key=lambda x: x[0])
        expected = list(range(len(items)))
        observed = [channel for channel, _ in items]
        if observed != expected:
            raise ValueError(
                f"Multi-image case '{identifier}' must use contiguous postfixes starting at 0000, got {observed}"
            )
        cases[identifier] = [path for _, path in items]
    if len(cases) == 0:
        raise ValueError(f"{label} does not contain any valid multi-image cases")
    return cases


def _scan_image_dir(folder: str, label: str, multi_image: bool) -> dict[str, list[str]]:
    return _scan_multi_image_dir(folder, label) if multi_image else _scan_single_image_dir(folder, label)


def _assert_matching_identifiers(
    left: dict[str, list[str]],
    right: dict[str, list[str]],
    left_label: str,
    right_label: str,
) -> list[str]:
    left_ids = set(left.keys())
    right_ids = set(right.keys())
    missing_in_right = sorted(left_ids.difference(right_ids))
    missing_in_left = sorted(right_ids.difference(left_ids))
    if missing_in_right or missing_in_left:
        pieces = []
        if missing_in_right:
            pieces.append(f"missing in {right_label}: {missing_in_right}")
        if missing_in_left:
            pieces.append(f"missing in {left_label}: {missing_in_left}")
        raise ValueError(
            f"{left_label} and {right_label} must contain the same case identifiers; " + "; ".join(pieces)
        )
    return sorted(left_ids)


def _preprocess_case(
    *,
    identifier: str,
    image_files: list[str],
    image_reader_name: str,
    image_dataset_json: Optional[dict],
    config: PreprocessingConfig,
    output_folder: Path,
    task_mode: str,
    run_stage: str,
    reference_files: Optional[list[str] | str] = None,
    reference_reader_name: Optional[str] = None,
    reference_dataset_json: Optional[dict] = None,
    storage_format: str = "blosc2",
    patch_size_hint: Optional[Sequence[int]] = None,
    patch_sampling_patch_sizes: Optional[dict[str, tuple[int, ...]]] = None,
    patch_sampling_threshold: Optional[float] = None,
    patch_sampling_min_fraction: float = 0.0,
    patch_sampling_source: str = "image",
    patch_sampling_max_starts: int = 8192,
) -> None:
    from .preprocessing import TaskAwarePreprocessor

    preprocessor = TaskAwarePreprocessor(config)
    image_reader = _build_reader(image_reader_name, image_files[0], image_dataset_json)
    reference_reader = None
    if reference_files is not None:
        reference_example = reference_files if isinstance(reference_files, str) else reference_files[0]
        reference_reader = _build_reader(
            reference_reader_name or image_reader_name,
            reference_example,
            reference_dataset_json or image_dataset_json,
        )
    case = preprocessor.run_task_case_from_files(
        image_files=image_files,
        image_reader=image_reader,
        task_mode=task_mode,
        run_stage=run_stage,
        reference_files=reference_files,
        reference_reader=reference_reader,
    )
    save_preprocessed_case(
        case,
        str(_prepare_output_prefix(output_folder, identifier)),
        storage_format=storage_format,
        patch_size_hint=patch_size_hint,
        patch_sampling_patch_sizes=patch_sampling_patch_sizes,
        patch_sampling_threshold=patch_sampling_threshold,
        patch_sampling_min_fraction=patch_sampling_min_fraction,
        patch_sampling_source=patch_sampling_source,
        patch_sampling_max_starts=patch_sampling_max_starts,
    )


def _preprocess_case_worker(payload: dict) -> None:
    _preprocess_case(**payload)


def _preprocess_segmentation_or_self_supervised(
    args: argparse.Namespace,
    config: PreprocessingConfig,
    dataset_json: Optional[dict],
    default_patch_size: Optional[Sequence[int]],
    default_configuration: Optional[str],
    configurations: Optional[dict],
) -> str:
    images = _scan_image_dir(args.images_dir, "--images-dir", args.multi_image)
    patch_sampling_patch_sizes = _build_patch_sampling_patch_sizes(default_patch_size, configurations)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.task_mode == TaskMode.SELF_SUPERVISED:
        if args.labels_dir is not None:
            raise ValueError("self_supervised does not accept --labels-dir")
        identifiers = sorted(images.keys())
        work_items = [
            {
                "identifier": identifier,
                "image_files": images[identifier],
                "image_reader_name": args.image_reader,
                "image_dataset_json": dataset_json,
                "config": config,
                "output_folder": output_folder,
                "task_mode": TaskMode.SELF_SUPERVISED,
                "run_stage": args.run_stage,
                "storage_format": args.storage_format,
                "patch_size_hint": default_patch_size,
                "patch_sampling_patch_sizes": patch_sampling_patch_sizes,
                "patch_sampling_threshold": args.patch_foreground_threshold,
                "patch_sampling_min_fraction": args.patch_foreground_min_fraction,
                "patch_sampling_source": args.patch_foreground_source,
                "patch_sampling_max_starts": args.patch_foreground_max_starts,
            }
            for identifier in identifiers
        ]
        _run_case_progress("cases", work_items, _preprocess_case_worker, args.num_processes)
        return save_preprocessed_dataset(
            folder=str(output_folder),
            task_mode=TaskMode.SELF_SUPERVISED,
            run_stage=args.run_stage,
            config=config,
            default_patch_size=default_patch_size,
            default_configuration=default_configuration,
            configurations=configurations,
            identifiers=identifiers,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            storage_format=args.storage_format,
        )

    if args.run_stage in {RunStage.TRAIN, RunStage.PREDICT_AND_EVALUATE}:
        if args.labels_dir is None:
            raise ValueError("segmentation train/predict_and_evaluate requires --labels-dir")
    if args.run_stage == RunStage.PREDICT and args.labels_dir is not None:
        raise ValueError("segmentation predict does not accept --labels-dir")

    labels = _scan_single_image_dir(args.labels_dir, "--labels-dir") if args.labels_dir is not None else None
    identifiers = sorted(images.keys()) if labels is None else _assert_matching_identifiers(images, labels, "images", "labels")
    work_items = [
        {
            "identifier": identifier,
            "image_files": images[identifier],
            "image_reader_name": args.image_reader,
            "image_dataset_json": dataset_json,
            "config": config,
            "output_folder": output_folder,
            "task_mode": TaskMode.SEGMENTATION,
            "run_stage": args.run_stage,
            "reference_files": None if labels is None else labels[identifier][0],
            "reference_reader_name": args.reference_reader,
            "reference_dataset_json": dataset_json,
            "storage_format": args.storage_format,
            "patch_size_hint": default_patch_size,
            "patch_sampling_patch_sizes": patch_sampling_patch_sizes,
            "patch_sampling_threshold": args.patch_foreground_threshold,
            "patch_sampling_min_fraction": args.patch_foreground_min_fraction,
            "patch_sampling_source": args.patch_foreground_source,
            "patch_sampling_max_starts": args.patch_foreground_max_starts,
        }
        for identifier in identifiers
    ]
    _run_case_progress("cases", work_items, _preprocess_case_worker, args.num_processes)
    return save_preprocessed_dataset(
        folder=str(output_folder),
        task_mode=TaskMode.SEGMENTATION,
        run_stage=args.run_stage,
        config=config,
        default_patch_size=default_patch_size,
        default_configuration=default_configuration,
        configurations=configurations,
        identifiers=identifiers,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        storage_format=args.storage_format,
    )


def _preprocess_paired(
    args: argparse.Namespace,
    config: PreprocessingConfig,
    dataset_json: Optional[dict],
    default_patch_size: Optional[Sequence[int]],
    default_configuration: Optional[str],
    configurations: Optional[dict],
) -> str:
    if args.source_dir is None:
        raise ValueError("paired_generative requires --source-dir")
    sources = _scan_image_dir(args.source_dir, "--source-dir", args.multi_image)
    patch_sampling_patch_sizes = _build_patch_sampling_patch_sizes(default_patch_size, configurations)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.run_stage in {RunStage.TRAIN, RunStage.PREDICT_AND_EVALUATE}:
        if args.target_dir is None:
            raise ValueError("paired_generative train/predict_and_evaluate requires --target-dir")
    if args.run_stage == RunStage.PREDICT and args.target_dir is not None:
        raise ValueError("paired_generative predict does not accept --target-dir")

    targets = _scan_image_dir(args.target_dir, "--target-dir", args.multi_image) if args.target_dir is not None else None
    identifiers = (
        sorted(sources.keys())
        if targets is None
        else _assert_matching_identifiers(sources, targets, "source", "target")
    )
    work_items = [
        {
            "identifier": identifier,
            "image_files": sources[identifier],
            "image_reader_name": args.source_reader,
            "image_dataset_json": dataset_json,
            "config": config,
            "output_folder": output_folder,
            "task_mode": TaskMode.PAIRED_GENERATIVE,
            "run_stage": args.run_stage,
            "reference_files": None if targets is None else targets[identifier],
            "reference_reader_name": args.target_reader,
            "reference_dataset_json": dataset_json,
            "storage_format": args.storage_format,
            "patch_size_hint": default_patch_size,
            "patch_sampling_patch_sizes": patch_sampling_patch_sizes,
            "patch_sampling_threshold": args.patch_foreground_threshold,
            "patch_sampling_min_fraction": args.patch_foreground_min_fraction,
            "patch_sampling_source": args.patch_foreground_source,
            "patch_sampling_max_starts": args.patch_foreground_max_starts,
        }
        for identifier in identifiers
    ]
    _run_case_progress("cases", work_items, _preprocess_case_worker, args.num_processes)
    return save_preprocessed_dataset(
        folder=str(output_folder),
        task_mode=TaskMode.PAIRED_GENERATIVE,
        run_stage=args.run_stage,
        config=config,
        default_patch_size=default_patch_size,
        default_configuration=default_configuration,
        configurations=configurations,
        identifiers=identifiers,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        storage_format=args.storage_format,
    )


def _preprocess_unpaired(
    args: argparse.Namespace,
    config_a: PreprocessingConfig,
    config_b: PreprocessingConfig,
    dataset_json_a: Optional[dict],
    dataset_json_b: Optional[dict],
    default_patch_size: Optional[Sequence[int]],
    default_configuration: Optional[str],
    configurations: Optional[dict],
) -> str:
    if args.domain_a_dir is None or args.domain_b_dir is None:
        raise ValueError("unpaired_generative requires --domain-a-dir and --domain-b-dir")
    if args.run_stage == RunStage.PREDICT_AND_EVALUATE:
        raise ValueError("unpaired_generative does not support predict_and_evaluate")

    domain_a = _scan_image_dir(args.domain_a_dir, "--domain-a-dir", args.multi_image)
    domain_b = _scan_image_dir(args.domain_b_dir, "--domain-b-dir", args.multi_image)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    folder_a = output_folder / args.folder_a_name
    folder_b = output_folder / args.folder_b_name
    folder_a.mkdir(exist_ok=True)
    folder_b.mkdir(exist_ok=True)
    patch_sampling_patch_sizes = _build_patch_sampling_patch_sizes(default_patch_size, configurations)

    identifiers_a = sorted(domain_a.keys())
    identifiers_b = sorted(domain_b.keys())
    work_items_a = [
        {
            "identifier": identifier,
            "image_files": domain_a[identifier],
            "image_reader_name": args.domain_a_reader,
            "image_dataset_json": dataset_json_a,
            "config": config_a,
            "output_folder": folder_a,
            "task_mode": TaskMode.UNPAIRED_GENERATIVE,
            "run_stage": args.run_stage,
            "storage_format": args.storage_format,
            "patch_size_hint": default_patch_size,
            "patch_sampling_patch_sizes": patch_sampling_patch_sizes,
            "patch_sampling_threshold": args.patch_foreground_threshold,
            "patch_sampling_min_fraction": args.patch_foreground_min_fraction,
            "patch_sampling_source": args.patch_foreground_source,
            "patch_sampling_max_starts": args.patch_foreground_max_starts,
        }
        for identifier in identifiers_a
    ]
    work_items_b = [
        {
            "identifier": identifier,
            "image_files": domain_b[identifier],
            "image_reader_name": args.domain_b_reader,
            "image_dataset_json": dataset_json_b,
            "config": config_b,
            "output_folder": folder_b,
            "task_mode": TaskMode.UNPAIRED_GENERATIVE,
            "run_stage": args.run_stage,
            "storage_format": args.storage_format,
            "patch_size_hint": default_patch_size,
            "patch_sampling_patch_sizes": patch_sampling_patch_sizes,
            "patch_sampling_threshold": args.patch_foreground_threshold,
            "patch_sampling_min_fraction": args.patch_foreground_min_fraction,
            "patch_sampling_source": args.patch_foreground_source,
            "patch_sampling_max_starts": args.patch_foreground_max_starts,
        }
        for identifier in identifiers_b
    ]
    _run_case_progress("domain A", work_items_a, _preprocess_case_worker, args.num_processes)
    _run_case_progress("domain B", work_items_b, _preprocess_case_worker, args.num_processes)

    return save_preprocessed_dataset(
        folder=str(output_folder),
        task_mode=TaskMode.UNPAIRED_GENERATIVE,
        run_stage=args.run_stage,
        default_patch_size=default_patch_size,
        default_configuration=default_configuration,
        configurations=configurations,
        folder_a=args.folder_a_name,
        folder_b=args.folder_b_name,
        config_a=config_a,
        config_b=config_b,
        identifiers_a=identifiers_a,
        identifiers_b=identifiers_b,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        storage_format=args.storage_format,
    )


def _preprocess_dataset_command(args: argparse.Namespace) -> int:
    total_steps = 4
    _ensure_storage_runtime(args.storage_format)
    _log_stage(1, total_steps, "Scan dataset", f"task_mode={args.task_mode}")
    base_config = _load_config(args.config_json, args.plans_file, args.configuration_name)
    config_a = _load_config(args.config_a_json, args.plans_a_file, args.configuration_a_name)
    config_b = _load_config(args.config_b_json, args.plans_b_file, args.configuration_b_name)
    default_patch_size = tuple(args.default_patch_size) if args.default_patch_size is not None else None
    default_configuration = None
    configurations = None

    if args.task_mode == TaskMode.UNPAIRED_GENERATIVE:
        domain_a = _scan_image_dir(args.domain_a_dir, "--domain-a-dir", args.multi_image)
        domain_b = _scan_image_dir(args.domain_b_dir, "--domain-b-dir", args.multi_image)
        _log_stage(2, total_steps, "Plan preprocessing", f"domain_a={len(domain_a)} cases, domain_b={len(domain_b)} cases")
        dataset_json_a = _discover_dataset_json(args.domain_a_dir)
        dataset_json_b = _discover_dataset_json(args.domain_b_dir)
        if base_config is not None:
            config_a = config_a or base_config
            config_b = config_b or base_config
        if config_a is None:
            config_a, fingerprint_a = _plan_config_from_cases(
                domain_a,
                args.domain_a_reader,
                dataset_json=dataset_json_a,
                num_processes=args.num_processes,
            )
            configurations_a = fingerprint_a.get("planning_configurations")
        else:
            configurations_a = None
        if config_b is None:
            config_b, fingerprint_b = _plan_config_from_cases(
                domain_b,
                args.domain_b_reader,
                dataset_json=dataset_json_b,
                num_processes=args.num_processes,
            )
            configurations_b = fingerprint_b.get("planning_configurations")
        else:
            configurations_b = None
        configurations = _merge_unpaired_configurations(configurations_a, configurations_b)
        default_configuration = _resolve_default_configuration(configurations)
        if default_patch_size is None and default_configuration is not None:
            default_patch_size = tuple(configurations[default_configuration]["patch_size"])
        _log_stage(3, total_steps, "Preprocess cases", "writing preprocessed domain folders")
        manifest_file = _preprocess_unpaired(
            args,
            config_a,
            config_b,
            dataset_json_a,
            dataset_json_b,
            default_patch_size,
            default_configuration,
            configurations,
        )
    else:
        if args.task_mode in {TaskMode.SEGMENTATION, TaskMode.SELF_SUPERVISED}:
            if args.images_dir is None:
                raise ValueError(f"{args.task_mode} requires --images-dir")
            dataset_json = _discover_dataset_json(args.images_dir, args.labels_dir)
            images = _scan_image_dir(args.images_dir, "--images-dir", args.multi_image)
            labels = (
                _scan_single_image_dir(args.labels_dir, "--labels-dir")
                if args.task_mode == TaskMode.SEGMENTATION
                and args.run_stage in {RunStage.TRAIN, RunStage.PREDICT_AND_EVALUATE}
                and args.labels_dir is not None
                else None
            )
            if labels is not None:
                _assert_matching_identifiers(images, labels, "images", "labels")
            _log_stage(
                2,
                total_steps,
                "Plan preprocessing",
                f"cases={len(images)}" + (", using dataset.json" if dataset_json is not None else ""),
            )
            if base_config is None:
                base_config, fingerprint = _plan_config_from_cases(
                    images,
                    args.image_reader,
                    dataset_json=dataset_json,
                    reference_cases=labels,
                    num_processes=args.num_processes,
                )
                configurations = fingerprint.get("planning_configurations")
                default_configuration = _resolve_default_configuration(configurations)
                if default_patch_size is None and default_configuration is not None:
                    default_patch_size = tuple(configurations[default_configuration]["patch_size"])
            _log_stage(3, total_steps, "Preprocess cases", "writing preprocessed case files")
            manifest_file = _preprocess_segmentation_or_self_supervised(
                args,
                base_config,
                dataset_json,
                default_patch_size,
                default_configuration,
                configurations,
            )
        elif args.task_mode == TaskMode.PAIRED_GENERATIVE:
            if args.source_dir is None:
                raise ValueError("paired_generative requires --source-dir")
            dataset_json = _discover_dataset_json(args.source_dir, args.target_dir)
            sources = _scan_image_dir(args.source_dir, "--source-dir", args.multi_image)
            targets = (
                _scan_image_dir(args.target_dir, "--target-dir", args.multi_image)
                if args.target_dir is not None and args.run_stage in {RunStage.TRAIN, RunStage.PREDICT_AND_EVALUATE}
                else None
            )
            if targets is not None:
                _assert_matching_identifiers(sources, targets, "source", "target")
            _log_stage(
                2,
                total_steps,
                "Plan preprocessing",
                f"cases={len(sources)}" + (", using dataset.json" if dataset_json is not None else ""),
            )
            if base_config is None:
                base_config, fingerprint = _plan_config_from_cases(
                    sources,
                    args.source_reader,
                    dataset_json=dataset_json,
                    num_processes=args.num_processes,
                )
                configurations = fingerprint.get("planning_configurations")
                default_configuration = _resolve_default_configuration(configurations)
                if default_patch_size is None and default_configuration is not None:
                    default_patch_size = tuple(configurations[default_configuration]["patch_size"])
            _log_stage(3, total_steps, "Preprocess cases", "writing preprocessed case files")
            manifest_file = _preprocess_paired(
                args,
                base_config,
                dataset_json,
                default_patch_size,
                default_configuration,
                configurations,
            )
        else:
            raise ValueError(f"Unsupported task_mode '{args.task_mode}'")

    _log_stage(4, total_steps, "Write manifest", Path(manifest_file).name)
    print(Path(manifest_file).resolve())
    return 0


def _save_dataset_command(args: argparse.Namespace) -> int:
    _ensure_storage_runtime(args.storage_format)
    config = _load_config(args.config_json, args.plans_file, args.configuration_name)
    config_a = _load_config(args.config_a_json, args.plans_a_file, args.configuration_a_name)
    config_b = _load_config(args.config_b_json, args.plans_b_file, args.configuration_b_name)

    manifest_file = save_preprocessed_dataset(
        folder=args.folder,
        task_mode=args.task_mode,
        run_stage=args.run_stage,
        config=config,
        default_patch_size=args.default_patch_size,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        folder_a=args.folder_a,
        folder_b=args.folder_b,
        config_a=config_a,
        config_b=config_b,
        random_pairing=not args.disable_random_pairing,
        storage_format=args.storage_format,
    )
    print(Path(manifest_file).resolve())
    return 0


def _show_manifest_command(args: argparse.Namespace) -> int:
    manifest = load_preprocessed_dataset_manifest(args.folder)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _build_config_argument_group(
    parser: argparse.ArgumentParser,
    *,
    title: str,
    json_flag: str,
    plans_flag: str,
    configuration_flag: str,
    label: str,
) -> argparse._ArgumentGroup:
    group = parser.add_argument_group(title)
    group.add_argument(
        json_flag,
        default=None,
        help=f"{label}에 사용할 PreprocessingConfig JSON. 지정하면 자동 planning 대신 이 값을 사용",
    )
    group.add_argument(
        plans_flag,
        default=None,
        help=f"{label}에 사용할 nnU-Net plans JSON",
    )
    group.add_argument(
        configuration_flag,
        default=None,
        help=f"{plans_flag}와 함께 사용할 nnU-Net configuration 이름",
    )
    return group


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m medimg_preprocessor",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess-dataset",
        help="raw 디렉토리를 스캔해 전처리하고 preprocessing_manifest.json을 생성",
        description=(
            "task별 raw 디렉토리를 스캔하고 case를 자동 매칭한 뒤, "
            "전처리 결과와 preprocessing_manifest.json을 생성합니다."
        ),
        epilog=PREPROCESS_DATASET_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    preprocess_parser.add_argument(
        "--output-folder",
        required=True,
        help="전처리 결과와 preprocessing_manifest.json을 저장할 출력 폴더",
    )
    preprocess_parser.add_argument(
        "--task-mode",
        required=True,
        choices=(
            TaskMode.SEGMENTATION,
            TaskMode.PAIRED_GENERATIVE,
            TaskMode.UNPAIRED_GENERATIVE,
            TaskMode.SELF_SUPERVISED,
        ),
        help="전처리할 task mode",
    )
    preprocess_parser.add_argument(
        "--run-stage",
        default=RunStage.TRAIN,
        choices=(RunStage.TRAIN, RunStage.PREDICT, RunStage.PREDICT_AND_EVALUATE),
        help="전처리 실행 단계. 기본값: train",
    )
    preprocess_parser.add_argument(
        "--default-patch-size",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help="manifest에 기록할 patch size override. 생략하면 planner가 자동 결정",
    )
    preprocess_parser.add_argument(
        "--storage-format",
        choices=("blosc2", "npz"),
        default="blosc2",
        help="저장 포맷. 기본값: blosc2",
    )
    preprocess_parser.add_argument(
        "--num-processes",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="planning/preprocessing에 사용할 worker process 수. 기본값: 사용 가능한 CPU의 절반",
    )
    preprocess_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="train 단계에서 자동 validation split에 사용할 비율. 기본값: 0.2",
    )
    preprocess_parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="자동 train/val split에 사용할 random seed. 기본값: 42",
    )
    preprocess_parser.add_argument(
        "--multi-image",
        action="store_true",
        help=(
            "이미지 디렉토리를 multi-image case로 처리합니다. "
            "파일명은 case_0001_0000, case_0001_0001 규칙을 따라야 합니다. "
            "단일 채널 case_0001_0000 파일은 이 플래그 없이도 허용됩니다."
        ),
    )

    single_or_seg_group = preprocess_parser.add_argument_group("segmentation / self_supervised 옵션")
    single_or_seg_group.add_argument("--images-dir", default=None, help="입력 이미지 디렉토리")
    single_or_seg_group.add_argument(
        "--labels-dir",
        default=None,
        help="segmentation label 디렉토리. segmentation train/predict_and_evaluate에서 필요",
    )
    single_or_seg_group.add_argument(
        "--image-reader",
        default="auto",
        choices=READER_CHOICES,
        help="--images-dir에 사용할 reader. 기본값: auto",
    )
    single_or_seg_group.add_argument(
        "--reference-reader",
        default=None,
        choices=READER_CHOICES,
        help="--labels-dir에 사용할 reader. 기본값: --image-reader와 동일",
    )

    paired_group = preprocess_parser.add_argument_group("paired_generative 옵션")
    paired_group.add_argument("--source-dir", default=None, help="source 이미지 디렉토리")
    paired_group.add_argument(
        "--target-dir",
        default=None,
        help="target 이미지 디렉토리. paired_generative train/predict_and_evaluate에서 필요",
    )
    paired_group.add_argument(
        "--source-reader",
        default="auto",
        choices=READER_CHOICES,
        help="--source-dir에 사용할 reader. 기본값: auto",
    )
    paired_group.add_argument(
        "--target-reader",
        default=None,
        choices=READER_CHOICES,
        help="--target-dir에 사용할 reader. 기본값: --source-reader와 동일",
    )

    preprocess_domain_group = preprocess_parser.add_argument_group("unpaired_generative 옵션")
    preprocess_domain_group.add_argument("--domain-a-dir", default=None, help="domain A 디렉토리")
    preprocess_domain_group.add_argument("--domain-b-dir", default=None, help="domain B 디렉토리")
    preprocess_domain_group.add_argument(
        "--domain-a-reader",
        default="auto",
        choices=READER_CHOICES,
        help="--domain-a-dir에 사용할 reader. 기본값: auto",
    )
    preprocess_domain_group.add_argument(
        "--domain-b-reader",
        default="auto",
        choices=READER_CHOICES,
        help="--domain-b-dir에 사용할 reader. 기본값: auto",
    )
    preprocess_domain_group.add_argument(
        "--folder-a-name",
        default="domain_a",
        help="unpaired 저장 시 --output-folder 아래에 생성할 domain A 하위 폴더 이름",
    )
    preprocess_domain_group.add_argument(
        "--folder-b-name",
        default="domain_b",
        help="unpaired 저장 시 --output-folder 아래에 생성할 domain B 하위 폴더 이름",
    )

    _build_config_argument_group(
        preprocess_parser,
        title="shared config",
        json_flag="--config-json",
        plans_flag="--plans-file",
        configuration_flag="--configuration-name",
        label="the dataset",
    )
    _build_config_argument_group(
        preprocess_parser,
        title="unpaired domain A config",
        json_flag="--config-a-json",
        plans_flag="--plans-a-file",
        configuration_flag="--configuration-a-name",
        label="domain A",
    )
    _build_config_argument_group(
        preprocess_parser,
        title="unpaired domain B config",
        json_flag="--config-b-json",
        plans_flag="--plans-b-file",
        configuration_flag="--configuration-b-name",
        label="domain B",
    )
    preprocess_parser.add_argument(
        "--patch-foreground-threshold",
        type=float,
        default=-900.0,
        help="Foreground threshold in pre-normalization intensity space used to precompute valid patch starts.",
    )
    preprocess_parser.add_argument(
        "--patch-foreground-min-fraction",
        type=float,
        default=0.0,
        help="Minimum foreground fraction in a patch required for a start location to be stored. Set 0 to disable.",
    )
    preprocess_parser.add_argument(
        "--patch-foreground-source",
        choices=("image", "target"),
        default="image",
        help="Array used to build the foreground mask for patch sampling metadata.",
    )
    preprocess_parser.add_argument(
        "--patch-foreground-max-starts",
        type=int,
        default=8192,
        help="Maximum number of valid patch start locations to store per patch size.",
    )

    preprocess_parser.set_defaults(func=_preprocess_dataset_command)

    save_parser = subparsers.add_parser(
        "save-dataset",
        help="이미 저장된 전처리 case들에 대해 preprocessing_manifest.json만 생성",
        description=(
            "이미 저장된 전처리 case 파일들이 있을 때, "
            "preprocessing_manifest.json만 다시 생성합니다."
        ),
        epilog=SAVE_DATASET_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    save_parser.add_argument(
        "--folder",
        required=True,
        help=(
            "dataset 폴더. single-folder task에서는 case 파일이 있는 폴더이고, "
            "unpaired task에서는 preprocessing_manifest.json이 위치할 루트 폴더입니다."
        ),
    )
    save_parser.add_argument(
        "--task-mode",
        required=True,
        choices=(
            TaskMode.SEGMENTATION,
            TaskMode.PAIRED_GENERATIVE,
            TaskMode.UNPAIRED_GENERATIVE,
            TaskMode.SELF_SUPERVISED,
        ),
        help="manifest에 기록할 task mode",
    )
    save_parser.add_argument(
        "--run-stage",
        default=RunStage.TRAIN,
        choices=(RunStage.TRAIN, RunStage.PREDICT, RunStage.PREDICT_AND_EVALUATE),
        help="manifest에 기록할 run stage. 기본값: train",
    )
    save_parser.add_argument(
        "--default-patch-size",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help="manifest에 기록할 patch size. 예: --default-patch-size 96 96 96",
    )
    save_parser.add_argument(
        "--storage-format",
        choices=("blosc2", "npz"),
        default="blosc2",
        help="manifest에 기록할 저장 포맷. 기본값: blosc2",
    )
    save_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="train 단계 manifest에 기록할 validation split 비율. 기본값: 0.2",
    )
    save_parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="train 단계 manifest에 기록할 split seed. 기본값: 42",
    )
    domain_group = save_parser.add_argument_group("unpaired dataset 옵션")
    domain_group.add_argument(
        "--folder-a",
        default=None,
        help="상대 또는 절대 domain A 폴더. task_mode=unpaired_generative일 때만 필요",
    )
    domain_group.add_argument(
        "--folder-b",
        default=None,
        help="상대 또는 절대 domain B 폴더. task_mode=unpaired_generative일 때만 필요",
    )
    domain_group.add_argument(
        "--disable-random-pairing",
        action="store_true",
        help="unpaired manifest에 random_pairing=false를 기록",
    )
    _build_config_argument_group(
        save_parser,
        title="single-folder dataset config",
        json_flag="--config-json",
        plans_flag="--plans-file",
        configuration_flag="--configuration-name",
        label="the single-folder dataset",
    )
    _build_config_argument_group(
        save_parser,
        title="unpaired domain A config",
        json_flag="--config-a-json",
        plans_flag="--plans-a-file",
        configuration_flag="--configuration-a-name",
        label="domain A",
    )
    _build_config_argument_group(
        save_parser,
        title="unpaired domain B config",
        json_flag="--config-b-json",
        plans_flag="--plans-b-file",
        configuration_flag="--configuration-b-name",
        label="domain B",
    )
    save_parser.set_defaults(func=_save_dataset_command)

    show_parser = subparsers.add_parser(
        "show-manifest",
        help="preprocessing_manifest.json 내용을 보기 좋게 출력",
        description="전처리된 dataset 폴더에서 preprocessing_manifest.json을 읽어 출력합니다.",
        epilog=SHOW_MANIFEST_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    show_parser.add_argument(
        "--folder",
        required=True,
        help="preprocessing_manifest.json이 들어 있는 dataset 폴더",
    )
    show_parser.set_defaults(func=_show_manifest_command)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def medimg_preprocess(argv: Optional[Sequence[str]] = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


