from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import multiprocessing
from typing import Dict, Optional, Sequence
import warnings

import numpy as np

from .config import PreprocessingConfig, ResamplingConfig
from .geometry import compute_new_shape, create_nonzero_mask
from .normalization import (
    CTNormalization,
    NoNormalization,
    RGBTo01Normalization,
    RescaleTo01Normalization,
    ZScoreNormalization,
)


ANISO_THRESHOLD = 3.0
MIN_FEATURE_MAP_EDGE_LENGTH = 4
UNET_MIN_BATCH_SIZE = 2
MAX_DATASET_COVERED = 0.05
REFERENCE_BATCH_SIZE_2D = 12
REFERENCE_BATCH_SIZE_3D = 2
REFERENCE_PATCH_VOLUME_2D = 2048 ** 2
REFERENCE_PATCH_VOLUME_3D = 256 ** 3

_CHANNEL_NAME_TO_NORMALIZATION = {
    "ct": CTNormalization,
    "nonorm": NoNormalization,
    "zscore": ZScoreNormalization,
    "rescale_to_0_1": RescaleTo01Normalization,
    "rgb_to_0_1": RGBTo01Normalization,
}


def _clip_values(
    values: np.ndarray,
    *,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> np.ndarray:
    if clip_min is None and clip_max is None:
        return values
    lower = -np.inf if clip_min is None else float(clip_min)
    upper = np.inf if clip_max is None else float(clip_max)
    return np.clip(values, lower, upper)


@dataclass(frozen=True)
class PlanningConfiguration:
    name: str
    spacing: tuple[float, ...]
    median_shape: tuple[int, ...]
    patch_size: tuple[int, ...]
    recommended_batch_size: int


def _fail_validation(message: str) -> None:
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)


def _get_channel_names(dataset_json: Optional[dict], num_channels: int) -> list[str]:
    if dataset_json is None:
        return ["zscore"] * num_channels
    if "channel_names" in dataset_json:
        channel_names = dataset_json["channel_names"]
    elif "modality" in dataset_json:
        channel_names = dataset_json["modality"]
    else:
        return ["zscore"] * num_channels

    if not isinstance(channel_names, dict):
        _fail_validation("dataset.json field 'channel_names' or 'modality' must be an object")

    ordered = []
    for key in sorted(channel_names.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        ordered.append(str(channel_names[key]))
    if len(ordered) != num_channels:
        _fail_validation(
            f"dataset.json channel count {len(ordered)} does not match detected image channels {num_channels}"
        )
    return ordered


def _get_normalization_scheme_name(channel_name: str) -> str:
    normalizer_cls = _CHANNEL_NAME_TO_NORMALIZATION.get(channel_name.casefold(), ZScoreNormalization)
    return normalizer_cls.__name__


def _get_shape_must_be_divisible_by(net_numpool_per_axis: Sequence[int]) -> np.ndarray:
    return 2 ** np.array(net_numpool_per_axis)


def _pad_shape(shape: Sequence[int], must_be_divisible_by: Sequence[int]) -> np.ndarray:
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    elif len(must_be_divisible_by) != len(shape):
        _fail_validation(
            "must_be_divisible_by must either be a scalar or match shape dimensionality, "
            f"got {len(must_be_divisible_by)} and {len(shape)}"
        )
    new_shape = [shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i] for i in range(len(shape))]
    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shape[i] -= must_be_divisible_by[i]
    return np.array(new_shape).astype(int)


def _get_pool_and_conv_props(
    spacing: Sequence[float],
    patch_size: Sequence[int],
    min_feature_map_size: int,
    max_numpool: int,
) -> tuple[list[int], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[int, ...], np.ndarray]:
    dim = len(spacing)
    current_spacing = deepcopy(list(spacing))
    current_size = deepcopy(list(patch_size))

    pool_op_kernel_sizes = [[1] * dim]
    conv_kernel_sizes = []
    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    while True:
        valid_axes_for_pool = [i for i in range(dim) if current_size[i] >= 2 * min_feature_map_size]
        if len(valid_axes_for_pool) < 1:
            break

        spacings_of_axes = [current_spacing[i] for i in valid_axes_for_pool]
        min_spacing_of_valid = min(spacings_of_axes)
        valid_axes_for_pool = [i for i in valid_axes_for_pool if current_spacing[i] / min_spacing_of_valid < 2]
        valid_axes_for_pool = [i for i in valid_axes_for_pool if num_pool_per_axis[i] < max_numpool]

        if len(valid_axes_for_pool) == 1 and current_size[valid_axes_for_pool[0]] < 3 * min_feature_map_size:
            break
        if len(valid_axes_for_pool) < 1:
            break

        for d in range(dim):
            if kernel_size[d] == 3:
                continue
            if current_spacing[d] / min(current_spacing) < 2:
                kernel_size[d] = 3

        other_axes = [i for i in range(dim) if i not in valid_axes_for_pool]
        pool_kernel_sizes = [0] * dim
        for axis in valid_axes_for_pool:
            pool_kernel_sizes[axis] = 2
            num_pool_per_axis[axis] += 1
            current_spacing[axis] *= 2
            current_size[axis] = np.ceil(current_size[axis] / 2)
        for axis in other_axes:
            pool_kernel_sizes[axis] = 1

        pool_op_kernel_sizes.append(pool_kernel_sizes)
        conv_kernel_sizes.append(deepcopy(kernel_size))

    must_be_divisible_by = _get_shape_must_be_divisible_by(num_pool_per_axis)
    patch_size = _pad_shape(patch_size, must_be_divisible_by)
    conv_kernel_sizes.append([3] * dim)

    def _to_tuple(lst):
        return tuple(_to_tuple(i) if isinstance(i, list) else i for i in lst)

    return (
        num_pool_per_axis,
        _to_tuple(pool_op_kernel_sizes),
        _to_tuple(conv_kernel_sizes),
        tuple(int(i) for i in patch_size),
        must_be_divisible_by,
    )


def _estimate_patch_and_batch(
    spacing: Sequence[float],
    median_shape: Sequence[float],
    approximate_n_voxels_dataset: float,
) -> tuple[tuple[int, ...], int]:
    spacing = np.asarray(spacing, dtype=np.float64)
    median_shape = np.asarray(median_shape, dtype=np.float64)
    if len(spacing) not in (2, 3):
        _fail_validation(f"Only 2D and 3D planning are supported, got spacing with {len(spacing)} dims")

    tmp = 1 / spacing
    if len(spacing) == 3:
        initial_patch_size = [round(i) for i in tmp * (REFERENCE_PATCH_VOLUME_3D / np.prod(tmp)) ** (1 / 3)]
        reference_patch_volume = REFERENCE_PATCH_VOLUME_3D
        reference_batch_size = REFERENCE_BATCH_SIZE_3D
    else:
        initial_patch_size = [round(i) for i in tmp * (REFERENCE_PATCH_VOLUME_2D / np.prod(tmp)) ** (1 / 2)]
        reference_patch_volume = REFERENCE_PATCH_VOLUME_2D
        reference_batch_size = REFERENCE_BATCH_SIZE_2D

    initial_patch_size = np.array([min(i, j) for i, j in zip(initial_patch_size, median_shape[: len(spacing)])])
    _, _, _, patch_size, shape_must_be_divisible_by = _get_pool_and_conv_props(
        spacing,
        initial_patch_size,
        MIN_FEATURE_MAP_EDGE_LENGTH,
        999999,
    )

    patch_size = np.asarray(patch_size, dtype=int)
    target_patch_volume = reference_patch_volume if len(spacing) == 2 else max(REFERENCE_PATCH_VOLUME_3D // 4, 128 ** 3)
    while np.prod(patch_size, dtype=np.int64) > target_patch_volume and np.any(patch_size > shape_must_be_divisible_by):
        axis_to_be_reduced = int(np.argsort([i / j for i, j in zip(patch_size, median_shape[: len(spacing)])])[-1])
        tmp_patch = list(patch_size)
        tmp_patch[axis_to_be_reduced] -= int(shape_must_be_divisible_by[axis_to_be_reduced])
        if tmp_patch[axis_to_be_reduced] < 2 * MIN_FEATURE_MAP_EDGE_LENGTH:
            break
        _, _, _, _, shape_must_be_divisible_by = _get_pool_and_conv_props(
            spacing,
            tmp_patch,
            MIN_FEATURE_MAP_EDGE_LENGTH,
            999999,
        )
        tmp_patch[axis_to_be_reduced] -= int(shape_must_be_divisible_by[axis_to_be_reduced])
        if tmp_patch[axis_to_be_reduced] < 2 * MIN_FEATURE_MAP_EDGE_LENGTH:
            break
        _, _, _, patch_size, shape_must_be_divisible_by = _get_pool_and_conv_props(
            spacing,
            tmp_patch,
            MIN_FEATURE_MAP_EDGE_LENGTH,
            999999,
        )
        patch_size = np.asarray(patch_size, dtype=int)

    patch_voxels = max(1, int(np.prod(patch_size, dtype=np.int64)))
    batch_size = round((reference_patch_volume / patch_voxels) * reference_batch_size)
    bs_corresponding_to_5_percent = round(approximate_n_voxels_dataset * MAX_DATASET_COVERED / float(patch_voxels))
    batch_size = max(min(batch_size, bs_corresponding_to_5_percent), UNET_MIN_BATCH_SIZE)
    return tuple(int(i) for i in patch_size), int(batch_size)


def collect_foreground_intensities(
    segmentation: np.ndarray,
    images: np.ndarray,
    seed: int = 1234,
    num_samples: int = 10000,
) -> tuple[list[np.ndarray], list[dict]]:
    if images.ndim != 4 or segmentation.ndim != 4:
        _fail_validation(
            f"collect_foreground_intensities expects image/seg with ndim=4, got {images.shape} and {segmentation.shape}"
        )
    if np.any(np.isnan(segmentation)) or np.any(np.isnan(images)):
        _fail_validation("NaN values are not allowed in images or segmentations")

    rs = np.random.RandomState(seed)
    intensities_per_channel = []
    intensity_statistics_per_channel = []

    foreground_mask = segmentation[0] > 0
    percentiles = np.array((0.5, 50.0, 99.5))

    for i in range(len(images)):
        foreground_pixels = images[i][foreground_mask]
        num_fg = len(foreground_pixels)
        intensities_per_channel.append(
            rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else np.array([], dtype=np.float32)
        )

        mean = median = mini = maxi = percentile_99_5 = percentile_00_5 = np.nan
        if num_fg > 0:
            percentile_00_5, median, percentile_99_5 = np.percentile(foreground_pixels, percentiles)
            mean = np.mean(foreground_pixels)
            mini = np.min(foreground_pixels)
            maxi = np.max(foreground_pixels)

        intensity_statistics_per_channel.append(
            {
                "mean": mean,
                "median": median,
                "min": mini,
                "max": maxi,
                "percentile_99_5": percentile_99_5,
                "percentile_00_5": percentile_00_5,
            }
        )

    return intensities_per_channel, intensity_statistics_per_channel


def collect_nonzero_intensities(
    images: np.ndarray,
    mask: np.ndarray,
    seed: int = 1234,
    num_samples: int = 10000,
) -> tuple[list[np.ndarray], list[dict]]:
    rs = np.random.RandomState(seed)
    intensities_per_channel = []
    intensity_statistics_per_channel = []
    percentiles = np.array((0.5, 50.0, 99.5))

    for i in range(len(images)):
        pixels = images[i][mask]
        if len(pixels) == 0:
            pixels = images[i].ravel()
        sampled = rs.choice(pixels, num_samples, replace=True) if len(pixels) > 0 else np.array([], dtype=np.float32)
        intensities_per_channel.append(sampled)

        mean = median = mini = maxi = percentile_99_5 = percentile_00_5 = np.nan
        if len(pixels) > 0:
            percentile_00_5, median, percentile_99_5 = np.percentile(pixels, percentiles)
            mean = np.mean(pixels)
            mini = np.min(pixels)
            maxi = np.max(pixels)

        intensity_statistics_per_channel.append(
            {
                "mean": mean,
                "median": median,
                "min": mini,
                "max": maxi,
                "percentile_99_5": percentile_99_5,
                "percentile_00_5": percentile_00_5,
            }
        )

    return intensities_per_channel, intensity_statistics_per_channel


def _fingerprint_case_worker(payload: dict) -> tuple[tuple[int, ...], tuple[float, ...], list[np.ndarray], float]:
    reader_class = payload["reader_class"]
    reader = reader_class()
    identifier = payload["identifier"]
    images, properties_images = reader.read_images(tuple(payload["image_files"]))
    spacing = properties_images["spacing"]

    reference_cases = payload.get("reference_cases")
    num_samples = int(payload["num_samples"])
    if reference_cases is not None:
        reference = reference_cases[identifier]
        if isinstance(reference, str):
            segmentation, _ = reader.read_seg(reference)
        else:
            if len(reference) != 1:
                _fail_validation(f"Segmentation planning expects exactly one reference file per case, got {reference}")
            segmentation, _ = reader.read_seg(reference[0])
        foreground_intensities_per_channel, _ = collect_foreground_intensities(
            segmentation,
            images,
            num_samples=num_samples,
        )
    else:
        foreground_mask = create_nonzero_mask(images)
        foreground_intensities_per_channel, _ = collect_nonzero_intensities(
            images,
            foreground_mask,
            num_samples=num_samples,
        )

    shape = images.shape[1:]
    return (
        tuple(int(i) for i in shape),
        tuple(float(i) for i in spacing),
        foreground_intensities_per_channel,
        1.0,
    )


def extract_fingerprint_from_cases(
    cases: dict[str, list[str]],
    reader,
    *,
    reference_cases: Optional[dict[str, Sequence[str] | str]] = None,
    dataset_json: Optional[dict] = None,
    num_foreground_samples_total: int = int(10e7),
    ct_clip_min: Optional[float] = None,
    ct_clip_max: Optional[float] = None,
    num_processes: int = 1,
) -> dict:
    if len(cases) == 0:
        _fail_validation("extract_fingerprint_from_cases requires at least one case")

    num_samples_per_case = int(num_foreground_samples_total // len(cases))
    payloads = [
        {
            "identifier": identifier,
            "image_files": cases[identifier],
            "reference_cases": reference_cases,
            "reader_class": reader.__class__,
            "num_samples": num_samples_per_case,
        }
        for identifier in sorted(cases.keys())
    ]
    if num_processes > 1:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            results = pool.map(_fingerprint_case_worker, payloads)
    else:
        results = [_fingerprint_case_worker(payload) for payload in payloads]

    shapes = [r[0] for r in results]
    spacings = [r[1] for r in results]
    num_channels = len(results[0][2])
    stacked_intensities = [np.concatenate([r[2][i] for r in results]) for i in range(num_channels)]
    channel_names = _get_channel_names(dataset_json, num_channels)

    percentiles = np.array((0.5, 50.0, 99.5))
    intensity_statistics_per_channel: Dict[int, dict] = {}
    for i in range(num_channels):
        values = stacked_intensities[i]
        if _CHANNEL_NAME_TO_NORMALIZATION.get(channel_names[i].casefold(), ZScoreNormalization) is CTNormalization:
            values = _clip_values(values, clip_min=ct_clip_min, clip_max=ct_clip_max)
        if values.size == 0:
            intensity_statistics_per_channel[i] = {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "percentile_99_5": float("nan"),
                "percentile_00_5": float("nan"),
                "clip_min": None,
                "clip_max": None,
            }
            continue
        percentile_00_5, median, percentile_99_5 = np.percentile(values, percentiles)
        intensity_statistics_per_channel[i] = {
            "mean": float(np.mean(values)),
            "median": float(median),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "percentile_99_5": float(percentile_99_5),
            "percentile_00_5": float(percentile_00_5),
            "clip_min": None if ct_clip_min is None else float(ct_clip_min),
            "clip_max": None if ct_clip_max is None else float(ct_clip_max),
        }

    return {
        "spacings": spacings,
        "shapes": shapes,
        "foreground_intensity_properties_per_channel": intensity_statistics_per_channel,
        "median_relative_size": float(np.median([r[3] for r in results], 0)),
    }


def determine_fullres_target_spacing(
    fingerprint: dict,
    *,
    anisotropy_threshold: float = ANISO_THRESHOLD,
    overwrite_target_spacing: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if overwrite_target_spacing is not None:
        return np.array(overwrite_target_spacing, dtype=np.float64)

    spacings = np.vstack(fingerprint["spacings"])
    sizes = fingerprint["shapes"]
    target = np.percentile(spacings, 50, 0)

    if len(target) != 3:
        return target

    target_size = np.percentile(np.vstack(sizes), 50, 0)
    worst_spacing_axis = int(np.argmax(target))
    other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
    other_spacings = [target[i] for i in other_axes]
    other_sizes = [target_size[i] for i in other_axes]

    has_aniso_spacing = target[worst_spacing_axis] > (anisotropy_threshold * max(other_spacings))
    has_aniso_voxels = target_size[worst_spacing_axis] * anisotropy_threshold < min(other_sizes)

    if has_aniso_spacing and has_aniso_voxels:
        spacings_of_that_axis = spacings[:, worst_spacing_axis]
        target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
        if target_spacing_of_that_axis < max(other_spacings):
            target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
        target[worst_spacing_axis] = target_spacing_of_that_axis
    return target


def determine_transpose(
    target_spacing: Sequence[float],
    *,
    suppress_transpose: bool = False,
) -> tuple[list[int], list[int]]:
    if suppress_transpose or len(target_spacing) != 3:
        identity = list(range(len(target_spacing)))
        return identity, identity

    target_spacing = np.asarray(target_spacing)
    max_spacing_axis = int(np.argmax(target_spacing))
    remaining_axes = [i for i in list(range(3)) if i != max_spacing_axis]
    transpose_forward = [max_spacing_axis] + remaining_axes
    transpose_backward = [int(np.argwhere(np.array(transpose_forward) == i)[0][0]) for i in range(3)]
    return transpose_forward, transpose_backward


def determine_normalization_scheme_and_mask(
    dataset_json: Optional[dict],
    fingerprint: dict,
    num_channels: int,
) -> tuple[list[str], list[bool]]:
    channel_names = _get_channel_names(dataset_json, num_channels)
    normalization_schemes = [_get_normalization_scheme_name(channel_name) for channel_name in channel_names]

    if fingerprint["median_relative_size"] < (3 / 4.0):
        use_nonzero_mask_for_norm = [
            _CHANNEL_NAME_TO_NORMALIZATION.get(channel_name.casefold(), ZScoreNormalization)
            .leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true
            for channel_name in channel_names
        ]
    else:
        use_nonzero_mask_for_norm = [False] * len(channel_names)

    return normalization_schemes, use_nonzero_mask_for_norm


def _build_configurations(
    fingerprint: dict,
    transpose_forward: Sequence[int],
    *,
    target_spacing: Sequence[float],
) -> dict[str, PlanningConfiguration]:
    new_shapes = [
        compute_new_shape(shape, spacing, target_spacing)
        for spacing, shape in zip(fingerprint["spacings"], fingerprint["shapes"])
    ]
    new_median_shape = np.median(np.vstack(new_shapes), 0)
    new_median_shape_transposed = new_median_shape[np.asarray(transpose_forward)]
    fullres_spacing_transposed = np.asarray(target_spacing)[np.asarray(transpose_forward)]
    approximate_n_voxels_dataset = float(
        np.prod(new_median_shape_transposed, dtype=np.float64) * len(fingerprint["shapes"])
    )

    configurations: dict[str, PlanningConfiguration] = {}
    patch_size_3d, batch_size_3d = _estimate_patch_and_batch(
        fullres_spacing_transposed,
        new_median_shape_transposed,
        approximate_n_voxels_dataset,
    )
    configurations["3d"] = PlanningConfiguration(
        name="3d",
        spacing=tuple(float(i) for i in fullres_spacing_transposed),
        median_shape=tuple(int(round(i)) for i in new_median_shape_transposed),
        patch_size=patch_size_3d,
        recommended_batch_size=batch_size_3d,
    )

    if len(fullres_spacing_transposed) >= 2:
        spacing_2d = tuple(float(i) for i in fullres_spacing_transposed[-2:])
        median_shape_2d = tuple(float(i) for i in new_median_shape_transposed[-2:])
        patch_size_2d, batch_size_2d = _estimate_patch_and_batch(
            spacing_2d,
            median_shape_2d,
            approximate_n_voxels_dataset,
        )
        configurations["2d"] = PlanningConfiguration(
            name="2d",
            spacing=spacing_2d,
            median_shape=tuple(int(round(i)) for i in median_shape_2d),
            patch_size=patch_size_2d,
            recommended_batch_size=batch_size_2d,
        )
    return configurations


def plan_preprocessing_from_cases(
    cases: dict[str, list[str]],
    reader,
    *,
    dataset_json: Optional[dict] = None,
    reference_cases: Optional[dict[str, Sequence[str] | str]] = None,
    suppress_transpose: bool = False,
    overwrite_target_spacing: Optional[Sequence[float]] = None,
    ct_clip_min: Optional[float] = None,
    ct_clip_max: Optional[float] = None,
    num_processes: int = 1,
) -> tuple[PreprocessingConfig, dict]:
    fingerprint = extract_fingerprint_from_cases(
        cases,
        reader,
        reference_cases=reference_cases,
        dataset_json=dataset_json,
        ct_clip_min=ct_clip_min,
        ct_clip_max=ct_clip_max,
        num_processes=num_processes,
    )
    first_identifier = sorted(cases.keys())[0]
    num_channels = len(reader.read_images(tuple(cases[first_identifier]))[0])

    target_spacing = determine_fullres_target_spacing(
        fingerprint,
        anisotropy_threshold=ANISO_THRESHOLD,
        overwrite_target_spacing=overwrite_target_spacing,
    )
    transpose_forward, _ = determine_transpose(target_spacing, suppress_transpose=suppress_transpose)
    normalization_schemes, use_mask_for_norm = determine_normalization_scheme_and_mask(
        dataset_json,
        fingerprint,
        num_channels,
    )

    config = PreprocessingConfig(
        spacing=tuple(float(i) for i in target_spacing[transpose_forward]),
        transpose_forward=tuple(int(i) for i in transpose_forward),
        normalization_schemes=tuple(normalization_schemes),
        use_mask_for_norm=tuple(bool(i) for i in use_mask_for_norm),
        foreground_intensity_properties_per_channel={
            str(k): v for k, v in fingerprint["foreground_intensity_properties_per_channel"].items()
        },
        resampling=ResamplingConfig(
            image_order=3,
            image_order_z=0,
            label_order=1,
            label_order_z=0,
            force_separate_z=None,
            separate_z_anisotropy_threshold=ANISO_THRESHOLD,
        ),
    )
    fingerprint["planning_configurations"] = {
        name: {
            "spacing": list(configuration.spacing),
            "median_shape": list(configuration.median_shape),
            "patch_size": list(configuration.patch_size),
            "recommended_batch_size": int(configuration.recommended_batch_size),
        }
        for name, configuration in _build_configurations(
            fingerprint,
            transpose_forward,
            target_spacing=target_spacing,
        ).items()
    }
    return config, fingerprint
