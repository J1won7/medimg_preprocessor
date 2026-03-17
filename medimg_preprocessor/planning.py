from __future__ import annotations

from typing import Dict, Optional, Sequence
import warnings

import numpy as np

from .config import PreprocessingConfig, ResamplingConfig
from .geometry import crop_to_nonzero
from .normalization import (
    CTNormalization,
    NoNormalization,
    RGBTo01Normalization,
    RescaleTo01Normalization,
    ZScoreNormalization,
)


ANISO_THRESHOLD = 3.0

_CHANNEL_NAME_TO_NORMALIZATION = {
    "ct": CTNormalization,
    "nonorm": NoNormalization,
    "zscore": ZScoreNormalization,
    "rescale_to_0_1": RescaleTo01Normalization,
    "rgb_to_0_1": RGBTo01Normalization,
}


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


def extract_fingerprint_from_cases(
    cases: dict[str, list[str]],
    reader,
    *,
    reference_cases: Optional[dict[str, Sequence[str] | str]] = None,
    num_foreground_samples_total: int = int(10e7),
) -> dict:
    if len(cases) == 0:
        _fail_validation("extract_fingerprint_from_cases requires at least one case")

    num_samples_per_case = int(num_foreground_samples_total // len(cases))
    results = []

    for identifier in sorted(cases.keys()):
        images, properties_images = reader.read_images(tuple(cases[identifier]))
        spacing = properties_images["spacing"]
        if reference_cases is not None:
            reference = reference_cases[identifier]
            if isinstance(reference, str):
                segmentation, _ = reader.read_seg(reference)
            else:
                if len(reference) != 1:
                    _fail_validation(
                        f"Segmentation planning expects exactly one reference file per case, got {reference}"
                    )
                segmentation, _ = reader.read_seg(reference[0])
            data_cropped, seg_cropped, _ = crop_to_nonzero(images, segmentation)
            foreground_intensities_per_channel, _ = collect_foreground_intensities(
                seg_cropped,
                data_cropped,
                num_samples=num_samples_per_case,
            )
        else:
            data_cropped, mask_reference, _ = crop_to_nonzero(images, None)
            foreground_mask = mask_reference[0] >= 0
            foreground_intensities_per_channel, _ = collect_nonzero_intensities(
                data_cropped,
                foreground_mask,
                num_samples=num_samples_per_case,
            )

        shape_before_crop = images.shape[1:]
        shape_after_crop = data_cropped.shape[1:]
        relative_size_after_cropping = np.prod(shape_after_crop) / np.prod(shape_before_crop)
        results.append(
            (
                tuple(int(i) for i in shape_after_crop),
                tuple(float(i) for i in spacing),
                foreground_intensities_per_channel,
                float(relative_size_after_cropping),
            )
        )

    shapes_after_crop = [r[0] for r in results]
    spacings = [r[1] for r in results]
    num_channels = len(results[0][2])
    stacked_intensities = [np.concatenate([r[2][i] for r in results]) for i in range(num_channels)]

    percentiles = np.array((0.5, 50.0, 99.5))
    intensity_statistics_per_channel: Dict[int, dict] = {}
    for i in range(num_channels):
        values = stacked_intensities[i]
        if values.size == 0:
            intensity_statistics_per_channel[i] = {
                "mean": float("nan"),
                "median": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "percentile_99_5": float("nan"),
                "percentile_00_5": float("nan"),
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
        }

    return {
        "spacings": spacings,
        "shapes_after_crop": shapes_after_crop,
        "foreground_intensity_properties_per_channel": intensity_statistics_per_channel,
        "median_relative_size_after_cropping": float(np.median([r[3] for r in results], 0)),
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
    sizes = fingerprint["shapes_after_crop"]
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

    if fingerprint["median_relative_size_after_cropping"] < (3 / 4.0):
        use_nonzero_mask_for_norm = [
            _CHANNEL_NAME_TO_NORMALIZATION.get(channel_name.casefold(), ZScoreNormalization)
            .leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true
            for channel_name in channel_names
        ]
    else:
        use_nonzero_mask_for_norm = [False] * len(channel_names)

    return normalization_schemes, use_nonzero_mask_for_norm


def plan_preprocessing_from_cases(
    cases: dict[str, list[str]],
    reader,
    *,
    dataset_json: Optional[dict] = None,
    reference_cases: Optional[dict[str, Sequence[str] | str]] = None,
    suppress_transpose: bool = False,
    overwrite_target_spacing: Optional[Sequence[float]] = None,
) -> tuple[PreprocessingConfig, dict]:
    fingerprint = extract_fingerprint_from_cases(cases, reader, reference_cases=reference_cases)
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
    return config, fingerprint
