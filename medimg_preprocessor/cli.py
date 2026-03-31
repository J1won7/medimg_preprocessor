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
    python -m medimg_preprocessor preprocess-dataset --task-mode segmentation --images-dir raw/imagesTr --target-dir raw/labelsTr --output-folder preprocessed_seg

  Segmentation with multi-image inputs:
    python -m medimg_preprocessor preprocess-dataset --task-mode segmentation --images-dir raw/imagesTr --target-dir raw/labelsTr --output-folder preprocessed_seg --multi-image

  Paired generative:
    python -m medimg_preprocessor preprocess-dataset --task-mode paired_generative --images-dir raw/source --target-dir raw/target --output-folder preprocessed_paired

  Unpaired generative:
    python -m medimg_preprocessor preprocess-dataset --task-mode unpaired_generative --images-dir raw/domain_a --target-dir raw/domain_b --output-folder preprocessed_unpaired
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

NORMALIZATION_METHOD_CHOICES = (
    "auto",
    "CTNormalization",
    "ZScoreNormalization",
    "MinMaxClipNormalization",
)
MASK_THRESHOLD_UNSET = object()


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


def _parse_optional_float_arg(value: str) -> Optional[float]:
    lowered = str(value).strip().lower()
    if lowered in {"none", "null"}:
        return None
    return float(value)


def _resolve_mask_thresholds(args: argparse.Namespace) -> tuple[Optional[float], Optional[float]]:
    common = args.mask_threshold
    image = args.images_mask_threshold
    target = args.target_mask_threshold

    def _resolve(specific):
        if specific is not MASK_THRESHOLD_UNSET:
            return specific
        if common is not MASK_THRESHOLD_UNSET:
            return common
        return None

    return _resolve(image), _resolve(target)


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


def _get_channel_names_for_logging(dataset_json: Optional[dict], num_channels: int) -> list[str]:
    if dataset_json is None:
        return ["unknown"] * num_channels
    if "channel_names" in dataset_json:
        channel_names = dataset_json["channel_names"]
    elif "modality" in dataset_json:
        channel_names = dataset_json["modality"]
    else:
        return ["unknown"] * num_channels
    if not isinstance(channel_names, dict):
        return ["unknown"] * num_channels
    ordered = []
    for key in sorted(channel_names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        ordered.append(str(channel_names[key]))
    if len(ordered) != num_channels:
        return ["unknown"] * num_channels
    return ordered


def _log_normalization_summary(
    label: str,
    config: Optional[PreprocessingConfig],
    *,
    dataset_json: Optional[dict],
    normalization_method: str,
) -> None:
    if config is None:
        return
    channel_names = _get_channel_names_for_logging(dataset_json, len(config.normalization_schemes))
    if normalization_method == "auto":
        source = "auto"
        if dataset_json is None:
            source += " (dataset.json 없음, 기본값 z-score)"
        else:
            source += " (dataset.json channel_names/modality 기준)"
    else:
        source = f"manual override ({normalization_method})"

    print(f"  normalization[{label}] {source}", flush=True)
    for idx, scheme in enumerate(config.normalization_schemes):
        parts = [f"ch{idx}", f"name={channel_names[idx]}", f"scheme={scheme}", f"use_mask={config.use_mask_for_norm[idx]}"]
        stats = config.foreground_intensity_properties_per_channel.get(str(idx), {})
        if scheme == "CTNormalization":
            clip_min = stats.get("clip_min", stats.get("percentile_00_5"))
            clip_max = stats.get("clip_max", stats.get("percentile_99_5"))
            parts.append(f"clip=[{clip_min}, {clip_max}]")
        elif scheme == "MinMaxClipNormalization":
            parts.append(f"clip=[{stats.get('clip_min')}, {stats.get('clip_max')}]")
        print("   - " + ", ".join(parts), flush=True)


def _plan_config_from_cases(
    cases: dict[str, list[str]],
    reader_name: str,
    *,
    dataset_json: Optional[dict] = None,
    reference_cases: Optional[dict[str, str]] = None,
    ct_clip_min: Optional[float] = None,
    ct_clip_max: Optional[float] = None,
    num_processes: int = 1,
) -> tuple[PreprocessingConfig, dict]:
    first_identifier = sorted(cases.keys())[0]
    reader = _build_reader(reader_name, cases[first_identifier][0], dataset_json)
    return plan_preprocessing_from_cases(
        cases,
        reader,
        dataset_json=dataset_json,
        reference_cases=reference_cases,
        ct_clip_min=ct_clip_min,
        ct_clip_max=ct_clip_max,
        num_processes=num_processes,
    )


def _resolve_normalization_method(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    if name not in NORMALIZATION_METHOD_CHOICES:
        valid = ", ".join(NORMALIZATION_METHOD_CHOICES)
        raise ValueError(f"Unsupported --normalization-method '{name}'. Supported values: {valid}")
    return None if name == "auto" else str(name)


def _override_normalization_config(
    config: Optional[PreprocessingConfig],
    *,
    method: Optional[str],
    normalization_min: Optional[float],
    normalization_max: Optional[float],
) -> Optional[PreprocessingConfig]:
    if config is None:
        return None

    canonical_method = _resolve_normalization_method(method)
    if canonical_method is None:
        if (normalization_min is None) != (normalization_max is None):
            raise ValueError("Use both --normalization-min and --normalization-max together")
        if normalization_min is not None:
            raise ValueError(
                "--normalization-min/--normalization-max require "
                "--normalization-method MinMaxClipNormalization"
            )
        return config

    if canonical_method == "MinMaxClipNormalization":
        if (normalization_min is None) != (normalization_max is None):
            raise ValueError("Use both --normalization-min and --normalization-max together")
        if normalization_min is None:
            raise ValueError(
                "MinMaxClipNormalization requires both --normalization-min and --normalization-max"
            )
        if normalization_min >= normalization_max:
            raise ValueError("--normalization-min must be smaller than --normalization-max")
    elif normalization_min is not None or normalization_max is not None:
        raise ValueError(
            "--normalization-min/--normalization-max are only supported with "
            "--normalization-method MinMaxClipNormalization"
        )

    num_channels = len(config.normalization_schemes)
    use_mask_for_norm = (
        tuple(bool(i) for i in config.use_mask_for_norm)
        if canonical_method == "ZScoreNormalization"
        else tuple(False for _ in range(num_channels))
    )
    intensity_properties = {
        str(k): dict(v) for k, v in config.foreground_intensity_properties_per_channel.items()
    }
    if canonical_method == "MinMaxClipNormalization":
        for channel_idx in range(num_channels):
            channel_key = str(channel_idx)
            channel_properties = dict(intensity_properties.get(channel_key, {}))
            channel_properties["clip_min"] = float(normalization_min)
            channel_properties["clip_max"] = float(normalization_max)
            intensity_properties[channel_key] = channel_properties

    return PreprocessingConfig(
        spacing=config.spacing,
        transpose_forward=config.transpose_forward,
        normalization_schemes=tuple(canonical_method for _ in range(num_channels)),
        use_mask_for_norm=use_mask_for_norm,
        foreground_intensity_properties_per_channel=intensity_properties,
        resampling=config.resampling,
    )


def _override_resampling_config(
    config: Optional[PreprocessingConfig],
    *,
    label_order: Optional[int],
    label_order_z: Optional[int],
) -> Optional[PreprocessingConfig]:
    if config is None:
        return None
    if label_order is None and label_order_z is None:
        return config

    next_label_order = config.resampling.label_order if label_order is None else int(label_order)
    next_label_order_z = config.resampling.label_order_z if label_order_z is None else int(label_order_z)
    if next_label_order < 0 or next_label_order_z < 0:
        raise ValueError("--label-order and --label-order-z must be non-negative")

    return PreprocessingConfig(
        spacing=config.spacing,
        transpose_forward=config.transpose_forward,
        normalization_schemes=config.normalization_schemes,
        use_mask_for_norm=config.use_mask_for_norm,
        foreground_intensity_properties_per_channel={
            str(k): dict(v) for k, v in config.foreground_intensity_properties_per_channel.items()
        },
        resampling=ResamplingConfig(
            image_order=config.resampling.image_order,
            image_order_z=config.resampling.image_order_z,
            label_order=next_label_order,
            label_order_z=next_label_order_z,
            force_separate_z=config.resampling.force_separate_z,
            separate_z_anisotropy_threshold=config.resampling.separate_z_anisotropy_threshold,
        ),
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
    patch_sampling_min_fraction: float = 0.0,
    patch_sampling_max_starts: int = 8192,
    save_mask: Optional[bool] = None,
    image_mask_files: Optional[list[str] | str] = None,
    target_mask_files: Optional[list[str] | str] = None,
    mask_reader_name: Optional[str] = None,
    mask_mode: Optional[str] = None,
    image_mask_threshold: Optional[float] = None,
    target_mask_threshold: Optional[float] = None,
    mask_fill_holes: bool = True,
    mask_keep_largest_component: bool = True,
    mask_closing_iters: int = 1,
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
    mask_reader = None
    mask_example = None
    if image_mask_files is not None:
        mask_example = image_mask_files if isinstance(image_mask_files, str) else image_mask_files[0]
    elif target_mask_files is not None:
        mask_example = target_mask_files if isinstance(target_mask_files, str) else target_mask_files[0]
    if mask_example is not None:
        mask_reader = _build_reader(mask_reader_name or image_reader_name, mask_example, image_dataset_json)
    case = preprocessor.run_task_case_from_files(
        image_files=image_files,
        image_reader=image_reader,
        task_mode=task_mode,
        run_stage=run_stage,
        reference_files=reference_files,
        reference_reader=reference_reader,
        image_mask_files=image_mask_files,
        target_mask_files=target_mask_files,
        mask_reader=mask_reader,
        mask_mode=mask_mode,
        image_mask_threshold=image_mask_threshold,
        target_mask_threshold=target_mask_threshold,
        mask_fill_holes=mask_fill_holes,
        mask_keep_largest_component=mask_keep_largest_component,
        mask_closing_iters=mask_closing_iters,
    )
    save_preprocessed_case(
        case,
        str(_prepare_output_prefix(output_folder, identifier)),
        storage_format=storage_format,
        patch_size_hint=patch_size_hint,
        patch_sampling_patch_sizes=patch_sampling_patch_sizes,
        patch_sampling_min_fraction=patch_sampling_min_fraction,
        patch_sampling_max_starts=patch_sampling_max_starts,
        save_mask=save_mask,
    )


def _preprocess_case_worker(payload: dict) -> None:
    _preprocess_case(**payload)


def _scan_optional_mask_dir(folder: Optional[str], label: str) -> Optional[dict[str, list[str]]]:
    if folder is None:
        return None
    return _scan_single_image_dir(folder, label)


def _preprocess_segmentation_or_self_supervised(
    args: argparse.Namespace,
    config: PreprocessingConfig,
    dataset_json: Optional[dict],
    default_patch_size: Optional[Sequence[int]],
    default_configuration: Optional[str],
    configurations: Optional[dict],
) -> str:
    images = _scan_image_dir(args.images_dir, "--images-dir", args.multi_image)
    image_masks = _scan_optional_mask_dir(args.images_mask_dir, "--images-mask-dir")
    image_mask_threshold, target_mask_threshold = _resolve_mask_thresholds(args)
    patch_sampling_patch_sizes = _build_patch_sampling_patch_sizes(default_patch_size, configurations)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.task_mode == TaskMode.SELF_SUPERVISED:
        if args.target_dir is not None:
            raise ValueError("self_supervised does not accept --target-dir")
        if args.target_mask_dir is not None:
            raise ValueError("self_supervised does not accept --target-mask-dir")
        if target_mask_threshold is not None:
            raise ValueError("self_supervised does not accept --target-mask-threshold")
        if image_masks is not None:
            _assert_matching_identifiers(images, image_masks, "images", "images masks")
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
                "patch_sampling_min_fraction": args.patch_mask_min_fraction,
                "patch_sampling_max_starts": args.patch_mask_max_starts,
                "save_mask": args.save_mask,
                "image_mask_files": None if image_masks is None else image_masks[identifier][0],
                "mask_reader_name": args.mask_reader,
                "mask_mode": args.masking_mode,
                "image_mask_threshold": image_mask_threshold,
                "target_mask_threshold": None,
                "mask_fill_holes": args.mask_fill_holes,
                "mask_keep_largest_component": args.mask_keep_largest_component,
                "mask_closing_iters": args.mask_closing_iters,
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
        if args.target_dir is None:
            raise ValueError("segmentation train/predict_and_evaluate requires --target-dir")
    if args.run_stage == RunStage.PREDICT and args.target_dir is not None:
        raise ValueError("segmentation predict does not accept --target-dir")
    if args.masking_mode is not None:
        raise ValueError("segmentation uses the label-derived mask automatically and does not accept --masking-mode")
    if args.images_mask_dir is not None or args.target_mask_dir is not None:
        raise ValueError("segmentation does not accept external mask directories")
    if image_mask_threshold is not None or target_mask_threshold is not None:
        raise ValueError("segmentation does not accept mask threshold options")

    labels = _scan_single_image_dir(args.target_dir, "--target-dir") if args.target_dir is not None else None
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
            "patch_sampling_min_fraction": args.patch_mask_min_fraction,
            "patch_sampling_max_starts": args.patch_mask_max_starts,
            "save_mask": args.save_mask,
            "mask_mode": None,
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
    if args.images_dir is None:
        raise ValueError("paired_generative requires --images-dir")
    sources = _scan_image_dir(args.images_dir, "--images-dir", args.multi_image)
    source_masks = _scan_optional_mask_dir(args.images_mask_dir, "--images-mask-dir")
    target_masks = _scan_optional_mask_dir(args.target_mask_dir, "--target-mask-dir")
    image_mask_threshold, target_mask_threshold = _resolve_mask_thresholds(args)
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
    if source_masks is not None:
        _assert_matching_identifiers(sources, source_masks, "source", "source masks")
    if target_masks is not None:
        if targets is None:
            raise ValueError("paired_generative --target-mask-dir requires --target-dir")
        _assert_matching_identifiers(targets, target_masks, "target", "target masks")
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
            "patch_sampling_min_fraction": args.patch_mask_min_fraction,
            "patch_sampling_max_starts": args.patch_mask_max_starts,
            "save_mask": args.save_mask,
            "image_mask_files": None if source_masks is None else source_masks[identifier][0],
            "target_mask_files": None if target_masks is None else target_masks[identifier][0],
            "mask_reader_name": args.mask_reader,
            "mask_mode": args.masking_mode,
            "image_mask_threshold": image_mask_threshold,
            "target_mask_threshold": target_mask_threshold,
            "mask_fill_holes": args.mask_fill_holes,
            "mask_keep_largest_component": args.mask_keep_largest_component,
            "mask_closing_iters": args.mask_closing_iters,
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
    if args.images_dir is None or args.target_dir is None:
        raise ValueError("unpaired_generative requires --images-dir and --target-dir")
    if args.run_stage == RunStage.PREDICT_AND_EVALUATE:
        raise ValueError("unpaired_generative does not support predict_and_evaluate")

    domain_a = _scan_image_dir(args.images_dir, "--images-dir", args.multi_image)
    domain_b = _scan_image_dir(args.target_dir, "--target-dir", args.multi_image)
    masks_a = _scan_optional_mask_dir(args.images_mask_dir, "--images-mask-dir")
    masks_b = _scan_optional_mask_dir(args.target_mask_dir, "--target-mask-dir")
    image_mask_threshold, target_mask_threshold = _resolve_mask_thresholds(args)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    folder_a = output_folder / args.folder_a_name
    folder_b = output_folder / args.folder_b_name
    folder_a.mkdir(exist_ok=True)
    folder_b.mkdir(exist_ok=True)
    patch_sampling_patch_sizes = _build_patch_sampling_patch_sizes(default_patch_size, configurations)

    identifiers_a = sorted(domain_a.keys())
    identifiers_b = sorted(domain_b.keys())
    if masks_a is not None:
        _assert_matching_identifiers(domain_a, masks_a, "domain A", "domain A masks")
    if masks_b is not None:
        _assert_matching_identifiers(domain_b, masks_b, "domain B", "domain B masks")
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
            "patch_sampling_min_fraction": args.patch_mask_min_fraction,
            "patch_sampling_max_starts": args.patch_mask_max_starts,
            "save_mask": args.save_mask,
            "image_mask_files": None if masks_a is None else masks_a[identifier][0],
            "target_mask_files": None,
            "mask_reader_name": args.mask_reader,
            "mask_mode": args.masking_mode,
            "image_mask_threshold": image_mask_threshold,
            "target_mask_threshold": None,
            "mask_fill_holes": args.mask_fill_holes,
            "mask_keep_largest_component": args.mask_keep_largest_component,
            "mask_closing_iters": args.mask_closing_iters,
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
            "patch_sampling_min_fraction": args.patch_mask_min_fraction,
            "patch_sampling_max_starts": args.patch_mask_max_starts,
            "save_mask": args.save_mask,
            "image_mask_files": None if masks_b is None else masks_b[identifier][0],
            "target_mask_files": None,
            "mask_reader_name": args.mask_reader,
            "mask_mode": args.masking_mode,
            "image_mask_threshold": target_mask_threshold,
            "target_mask_threshold": None,
            "mask_fill_holes": args.mask_fill_holes,
            "mask_keep_largest_component": args.mask_keep_largest_component,
            "mask_closing_iters": args.mask_closing_iters,
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
    if (args.ct_clip_min is None) != (args.ct_clip_max is None):
        raise ValueError("Use both --ct-clip-min and --ct-clip-max together")
    if args.ct_clip_min is not None and args.ct_clip_min >= args.ct_clip_max:
        raise ValueError("--ct-clip-min must be smaller than --ct-clip-max")
    _resolve_normalization_method(args.normalization_method)
    _log_stage(1, total_steps, "Scan dataset", f"task_mode={args.task_mode}")
    base_config = _load_config(args.config_json, args.plans_file, args.configuration_name)
    config_a = _load_config(args.config_a_json, args.plans_a_file, args.configuration_a_name)
    config_b = _load_config(args.config_b_json, args.plans_b_file, args.configuration_b_name)
    default_patch_size = tuple(args.default_patch_size) if args.default_patch_size is not None else None
    default_configuration = None
    configurations = None

    if args.task_mode == TaskMode.UNPAIRED_GENERATIVE:
        if args.images_dir is None or args.target_dir is None:
            raise ValueError("unpaired_generative requires --images-dir and --target-dir")
        domain_a = _scan_image_dir(args.images_dir, "--images-dir", args.multi_image)
        domain_b = _scan_image_dir(args.target_dir, "--target-dir", args.multi_image)
        _log_stage(2, total_steps, "Plan preprocessing", f"domain_a={len(domain_a)} cases, domain_b={len(domain_b)} cases")
        dataset_json_a = _discover_dataset_json(args.images_dir)
        dataset_json_b = _discover_dataset_json(args.target_dir)
        if base_config is not None:
            config_a = config_a or base_config
            config_b = config_b or base_config
        if config_a is None:
            config_a, fingerprint_a = _plan_config_from_cases(
                domain_a,
                args.domain_a_reader,
                dataset_json=dataset_json_a,
                ct_clip_min=args.ct_clip_min,
                ct_clip_max=args.ct_clip_max,
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
                ct_clip_min=args.ct_clip_min,
                ct_clip_max=args.ct_clip_max,
                num_processes=args.num_processes,
            )
            configurations_b = fingerprint_b.get("planning_configurations")
        else:
            configurations_b = None
        config_a = _override_normalization_config(
            config_a,
            method=args.normalization_method,
            normalization_min=args.normalization_min,
            normalization_max=args.normalization_max,
        )
        config_a = _override_resampling_config(
            config_a,
            label_order=args.label_order,
            label_order_z=args.label_order_z,
        )
        config_b = _override_normalization_config(
            config_b,
            method=args.normalization_method,
            normalization_min=args.normalization_min,
            normalization_max=args.normalization_max,
        )
        config_b = _override_resampling_config(
            config_b,
            label_order=args.label_order,
            label_order_z=args.label_order_z,
        )
        _log_normalization_summary(
            "domain_a",
            config_a,
            dataset_json=dataset_json_a,
            normalization_method=args.normalization_method,
        )
        _log_normalization_summary(
            "domain_b",
            config_b,
            dataset_json=dataset_json_b,
            normalization_method=args.normalization_method,
        )
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
            dataset_json = _discover_dataset_json(args.images_dir, args.target_dir)
            images = _scan_image_dir(args.images_dir, "--images-dir", args.multi_image)
            labels = (
                _scan_single_image_dir(args.target_dir, "--target-dir")
                if args.task_mode == TaskMode.SEGMENTATION
                and args.run_stage in {RunStage.TRAIN, RunStage.PREDICT_AND_EVALUATE}
                and args.target_dir is not None
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
                    ct_clip_min=args.ct_clip_min,
                    ct_clip_max=args.ct_clip_max,
                    num_processes=args.num_processes,
                )
                configurations = fingerprint.get("planning_configurations")
                default_configuration = _resolve_default_configuration(configurations)
                if default_patch_size is None and default_configuration is not None:
                    default_patch_size = tuple(configurations[default_configuration]["patch_size"])
            base_config = _override_normalization_config(
                base_config,
                method=args.normalization_method,
                normalization_min=args.normalization_min,
                normalization_max=args.normalization_max,
            )
            base_config = _override_resampling_config(
                base_config,
                label_order=args.label_order,
                label_order_z=args.label_order_z,
            )
            _log_normalization_summary(
                "image",
                base_config,
                dataset_json=dataset_json,
                normalization_method=args.normalization_method,
            )
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
            if args.images_dir is None:
                raise ValueError("paired_generative requires --images-dir")
            dataset_json = _discover_dataset_json(args.images_dir, args.target_dir)
            sources = _scan_image_dir(args.images_dir, "--images-dir", args.multi_image)
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
                    ct_clip_min=args.ct_clip_min,
                    ct_clip_max=args.ct_clip_max,
                    num_processes=args.num_processes,
                )
                configurations = fingerprint.get("planning_configurations")
                default_configuration = _resolve_default_configuration(configurations)
                if default_patch_size is None and default_configuration is not None:
                    default_patch_size = tuple(configurations[default_configuration]["patch_size"])
            base_config = _override_normalization_config(
                base_config,
                method=args.normalization_method,
                normalization_min=args.normalization_min,
                normalization_max=args.normalization_max,
            )
            base_config = _override_resampling_config(
                base_config,
                label_order=args.label_order,
                label_order_z=args.label_order_z,
            )
            _log_normalization_summary(
                "source",
                base_config,
                dataset_json=dataset_json,
                normalization_method=args.normalization_method,
            )
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

    single_or_seg_group = preprocess_parser.add_argument_group("shared input / target 옵션")
    single_or_seg_group.add_argument("--images-dir", default=None, help="입력 이미지 디렉토리")
    single_or_seg_group.add_argument(
        "--target-dir",
        default=None,
        help=(
            "target 디렉토리. segmentation에서는 label, paired_generative에서는 paired target, "
            "unpaired_generative에서는 domain B로 사용됩니다."
        ),
    )
    single_or_seg_group.add_argument(
        "--label-dir",
        dest="target_dir",
        default=None,
        help="Deprecated alias for --target-dir in segmentation mode.",
    )
    single_or_seg_group.add_argument(
        "--source-dir",
        dest="images_dir",
        default=None,
        help="Deprecated alias for --images-dir in paired_generative mode.",
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
        help="--target-dir에 사용할 reader. 기본값: --image-reader와 동일",
    )

    paired_group = preprocess_parser.add_argument_group("paired_generative 옵션")
    paired_group.add_argument(
        "--source-reader",
        default="auto",
        choices=READER_CHOICES,
        help="--images-dir에 사용할 reader. 기본값: auto",
    )
    paired_group.add_argument(
        "--target-reader",
        default=None,
        choices=READER_CHOICES,
        help="--target-dir에 사용할 reader. 기본값: --source-reader와 동일",
    )

    preprocess_domain_group = preprocess_parser.add_argument_group("unpaired_generative 옵션")
    preprocess_domain_group.add_argument(
        "--domain-a-reader",
        default="auto",
        choices=READER_CHOICES,
        help="--images-dir(domain A)에 사용할 reader. 기본값: auto",
    )
    preprocess_domain_group.add_argument(
        "--domain-b-reader",
        default="auto",
        choices=READER_CHOICES,
        help="--target-dir(domain B)에 사용할 reader. 기본값: auto",
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

    masking_group = preprocess_parser.add_argument_group("masking options")
    masking_group.add_argument(
        "--masking-mode",
        choices=("threshold",),
        default=None,
        help=(
            "Optional rule for generating a patch sampling mask when no external mask directory is provided. "
            "Segmentation always uses the label-derived mask automatically. "
            "If omitted, non-segmentation modes sample from the full spatial extent."
        ),
    )
    masking_group.add_argument("--images-mask-dir", default=None, help="External mask directory aligned with --images-dir")
    masking_group.add_argument("--target-mask-dir", default=None, help="External mask directory aligned with --target-dir")
    masking_group.add_argument("--mask-reader", default="auto", choices=READER_CHOICES, help="Reader used for external masks. Default: auto")
    masking_group.add_argument(
        "--mask-threshold",
        type=_parse_optional_float_arg,
        default=MASK_THRESHOLD_UNSET,
        help="Common threshold shorthand for both image and target mask generation. Use 'none' to disable.",
    )
    masking_group.add_argument(
        "--images-mask-threshold",
        type=_parse_optional_float_arg,
        default=MASK_THRESHOLD_UNSET,
        help="Threshold for generating an image-side mask. Overrides --mask-threshold for the image side. Use 'none' to disable.",
    )
    masking_group.add_argument(
        "--target-mask-threshold",
        type=_parse_optional_float_arg,
        default=MASK_THRESHOLD_UNSET,
        help="Threshold for generating a target-side mask. Overrides --mask-threshold for the target side. Use 'none' to disable.",
    )
    masking_group.add_argument("--mask-fill-holes", action=argparse.BooleanOptionalAction, default=True, help="Fill holes in the generated mask. Default: true")
    masking_group.add_argument(
        "--mask-keep-largest-component",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only the largest connected component in the generated mask. Default: true",
    )
    masking_group.add_argument("--mask-closing-iters", type=int, default=1, help="Binary closing iterations applied to generated masks. Default: 1")
    masking_group.add_argument(
        "--patch-mask-min-fraction",
        type=float,
        default=0.5,
        help="Legacy fallback option for precomputed valid-start sampling. nnUNet-style dynamic mask-location sampling ignores this in newly preprocessed datasets.",
    )
    masking_group.add_argument(
        "--patch-mask-max-starts",
        type=int,
        default=8192,
        help="Maximum number of foreground/mask voxel locations to store for dynamic patch sampling.",
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
        "--ct-clip-min",
        type=float,
        default=None,
        help="Optional fixed lower HU/intensity clip for CT normalization. Default keeps percentile-based clipping.",
    )
    preprocess_parser.add_argument(
        "--ct-clip-max",
        type=float,
        default=None,
        help="Optional fixed upper HU/intensity clip for CT normalization. Default keeps percentile-based clipping.",
    )
    preprocess_parser.add_argument(
        "--normalization-method",
        default="auto",
        choices=NORMALIZATION_METHOD_CHOICES,
        help=(
            "Override normalization for all image channels. "
            "Supported values: auto, CTNormalization, "
            "ZScoreNormalization, MinMaxClipNormalization."
        ),
    )
    preprocess_parser.add_argument(
        "--normalization-min",
        type=float,
        default=None,
        help="Lower bound used by MinMaxClipNormalization. Requires --normalization-method MinMaxClipNormalization.",
    )
    preprocess_parser.add_argument(
        "--normalization-max",
        type=float,
        default=None,
        help="Upper bound used by MinMaxClipNormalization. Requires --normalization-method MinMaxClipNormalization.",
    )
    preprocess_parser.add_argument(
        "--label-order",
        type=int,
        default=None,
        help=(
            "Override the segmentation-label resampling order. "
            "0 keeps nearest-neighbor style label preservation. "
            "1 resizes each label mask separately and can smooth boundaries."
        ),
    )
    preprocess_parser.add_argument(
        "--label-order-z",
        type=int,
        default=None,
        help=(
            "Override the z-axis segmentation-label resampling order when separate-z resampling is used. "
            "0 is the safest choice for label-id preservation."
        ),
    )
    preprocess_parser.add_argument(
        "--save-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Store the derived sampling mask as a separate case file. "
            "By default this is disabled for segmentation train and enabled for other modes/stages."
        ),
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


