from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
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
from .geometry import resample_array
from .imageio import (
    NaturalImage2DIO,
    NibabelIO,
    NibabelIOWithReorient,
    SimpleITKIO,
    SimpleITKIOWithReorient,
    Tiff3DIO,
    determine_reader_writer_from_file_ending,
)
from .preprocessing import RunStage, TaskAwarePreprocessor, TaskMode


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


def _fail_validation(message: str) -> None:
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)


def _require_torch() -> None:
    if torch is None:
        _fail_validation("torch is required for inference datasets")


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


def _build_reader(reader_name: str, example_file: str):
    registry = {
        "nibabel": NibabelIO,
        "nibabel_reorient": NibabelIOWithReorient,
        "simpleitk": SimpleITKIO,
        "simpleitk_reorient": SimpleITKIOWithReorient,
        "tiff3d": Tiff3DIO,
        "natural_2d": NaturalImage2DIO,
    }
    if reader_name == "auto":
        ending = _detect_file_ending(example_file)
        return determine_reader_writer_from_file_ending(ending, example_file=example_file, verbose=False)()
    if reader_name not in registry:
        _fail_validation(f"Unsupported reader '{reader_name}'")
    return registry[reader_name]()


def _scan_image_dir(folder: str, multi_image: bool = False) -> dict[str, list[str]]:
    root = Path(folder)
    if not root.is_dir():
        _fail_validation(f"Image directory does not exist: {folder}")
    files = [path for path in root.rglob("*") if path.is_file() and _detect_file_ending(path.name) in SUPPORTED_SCAN_ENDINGS]
    if len(files) == 0:
        _fail_validation(f"No supported image files found in {folder}")
    grouped: dict[str, list[str]] = {}
    for file in sorted(files):
        stem = _strip_known_suffix(file.name)
        identifier = stem
        if multi_image:
            match = MULTI_IMAGE_PATTERN.match(stem)
            if match is None:
                _fail_validation(
                    "multi_image=True expects filenames like case_0001_0000.nii.gz; "
                    f"got '{file.name}'"
                )
            identifier = match.group("identifier")
        grouped.setdefault(identifier, []).append(str(file))
    for identifier, paths in grouped.items():
        grouped[identifier] = sorted(paths)
    return grouped


def _resolve_patch_size(array: np.ndarray, patch_size: Sequence[int], context: str) -> tuple[int, ...]:
    spatial_dims = array.ndim - 1
    if len(patch_size) == spatial_dims:
        return tuple(int(i) for i in patch_size)
    if spatial_dims == 3 and len(patch_size) == 2:
        return (1, int(patch_size[0]), int(patch_size[1]))
    _fail_validation(
        f"{context} expects patch_size with {spatial_dims} spatial dims, got {len(patch_size)} "
        f"for array shape {array.shape}"
    )


def _pad_to_patch_size(array: np.ndarray, patch_size: Sequence[int]) -> np.ndarray:
    pad_width = [(0, 0)]
    for current, wanted in zip(array.shape[1:], patch_size):
        missing = max(int(wanted) - int(current), 0)
        before = missing // 2
        after = missing - before
        pad_width.append((before, after))
    if any(p != (0, 0) for p in pad_width):
        array = np.pad(array, pad_width, mode="constant", constant_values=0)
    return array


def _crop_with_starts(array: np.ndarray, patch_size: Sequence[int], starts: Sequence[int]) -> np.ndarray:
    patch_size = tuple(int(i) for i in patch_size)
    if len(starts) != len(patch_size):
        _fail_validation("starts and patch_size must have the same dimensionality")
    array = _pad_to_patch_size(array, patch_size)
    slicer = (slice(None),) + tuple(slice(int(s), int(s) + int(p)) for s, p in zip(starts, patch_size))
    return array[slicer]


def _compute_sliding_starts(length: int, patch: int, overlap: float) -> list[int]:
    length = int(length)
    patch = int(patch)
    if length <= patch:
        return [0]
    step = max(1, int(round(patch * (1.0 - float(overlap)))))
    starts = list(range(0, max(length - patch, 0) + 1, step))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return [int(i) for i in starts]


def _compute_patch_starts(spatial_shape: Sequence[int], patch_size: Sequence[int], overlap: float) -> list[tuple[int, ...]]:
    axes = [_compute_sliding_starts(length, patch, overlap) for length, patch in zip(spatial_shape, patch_size)]
    return [tuple(int(axis_values[idx]) for axis_values, idx in zip(axes, indices)) for indices in np.ndindex(*(len(a) for a in axes))]


def _bbox_slices(bbox: Sequence[Sequence[int]]) -> tuple[slice, ...]:
    return tuple(slice(int(bounds[0]), int(bounds[1])) for bounds in bbox)


def _inverse_normalize_channel(image: np.ndarray, scheme: str, stats: dict) -> np.ndarray:
    eps = 1e-8
    image = image.astype(np.float32, copy=False)
    scheme_key = str(scheme).casefold()
    if scheme_key in {"nonorm", "nonormalization"}:
        return image
    if scheme_key in {"ct", "ctnormalization", "zscore", "zscorenormalization"}:
        if "mean" not in stats or "std" not in stats:
            _fail_validation(f"Cannot invert normalization scheme '{scheme}' without mean/std statistics")
        return image * max(float(stats["std"]), eps) + float(stats["mean"])
    if scheme_key in {"minmaxclip", "minmax_clip", "minmaxclipnormalization"}:
        clip_min = stats.get("clip_min")
        clip_max = stats.get("clip_max")
        if clip_min is None or clip_max is None:
            _fail_validation(f"Cannot invert normalization scheme '{scheme}' without clip_min/clip_max")
        return image * max(float(clip_max) - float(clip_min), eps) + float(clip_min)
    _fail_validation(
        f"Inverse normalization is not supported for scheme '{scheme}'. "
        "Use no normalization, CT/ZScore, or MinMaxClip for NIfTI export."
    )


def _inverse_normalize_prediction(pred: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    restored = pred.astype(np.float32, copy=True)
    for channel in range(restored.shape[0]):
        scheme = config.normalization_schemes[channel]
        stats = config.foreground_intensity_properties_per_channel.get(str(channel), {})
        restored[channel] = _inverse_normalize_channel(restored[channel], scheme, stats)
    return restored


def _undo_preprocessing(pred: np.ndarray, properties: Dict[str, Any], config: PreprocessingConfig) -> np.ndarray:
    restored = pred.astype(np.float32, copy=False)
    settings = properties.get("medimg_preprocessor_settings", {})

    if settings.get("resample", True):
        cropped_shape = tuple(int(i) for i in properties["shape_after_cropping_and_before_resampling"])
        restored = resample_array(
            restored,
            cropped_shape,
            properties["spacing_after_resampling"],
            properties["spacing_after_transpose"],
            is_seg=False,
            order=config.resampling.image_order,
            order_z=config.resampling.image_order_z,
            force_separate_z=config.resampling.force_separate_z,
            separate_z_anisotropy_threshold=config.resampling.separate_z_anisotropy_threshold,
        )

    if settings.get("crop_to_nonzero", True) and properties.get("bbox_used_for_cropping") is not None:
        shape_before_cropping = tuple(int(i) for i in properties["shape_before_cropping"])
        transposed_shape = tuple(shape_before_cropping[i] for i in config.transpose_forward)
        full = np.zeros((restored.shape[0], *transposed_shape), dtype=restored.dtype)
        full[(slice(None),) + _bbox_slices(properties["bbox_used_for_cropping"])] = restored
        restored = full

    if settings.get("transpose", True):
        inverse_axes = np.argsort(np.asarray(config.transpose_forward))
        restored = restored.transpose((0, *[int(i) + 1 for i in inverse_axes]))

    return restored


def _save_nifti_like_reference(volume: np.ndarray, reference_path: str | Path, output_path: str | Path) -> None:
    try:
        import nibabel as nib
    except ModuleNotFoundError:
        _fail_validation("nibabel is required to save NIfTI inference outputs")
    reference = nib.load(str(reference_path))
    data = volume[0].transpose((2, 1, 0)).astype(np.float32, copy=False)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data, affine=reference.affine, header=header)
    image.set_qform(reference.get_qform(), code=int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), code=int(reference.header["sform_code"]))
    nib.save(image, str(output_path))


@dataclass
class RawInferenceCase:
    identifier: str
    image_files: list[str]
    image: np.ndarray
    properties: dict
    patch_size: tuple[int, ...]
    patch_starts: list[tuple[int, ...]]


class InferencePatchAccumulator:
    def __init__(self, spatial_shape: Sequence[int], channels: int = 1):
        self.spatial_shape = tuple(int(i) for i in spatial_shape)
        self.channels = int(channels)
        self.value_sum = np.zeros((self.channels, *self.spatial_shape), dtype=np.float32)
        self.value_count = np.zeros((1, *self.spatial_shape), dtype=np.float32)

    def add_patch(self, patch: np.ndarray, starts: Sequence[int]) -> None:
        patch = np.asarray(patch, dtype=np.float32)
        if patch.ndim == len(self.spatial_shape):
            patch = patch[None]
        slices = tuple(slice(int(s), int(s) + int(size)) for s, size in zip(starts, patch.shape[1:]))
        self.value_sum[(slice(None),) + slices] += patch
        self.value_count[(slice(None),) + slices] += 1.0

    def finalize(self) -> np.ndarray:
        return self.value_sum / np.clip(self.value_count, 1e-8, None)


class RawInferencePatchDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        config: PreprocessingConfig,
        patch_size: Sequence[int],
        *,
        overlap: float = 0.5,
        image_reader: str = "auto",
        multi_image: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        _require_torch()
        if patch_size is None or len(tuple(patch_size)) == 0:
            _fail_validation("RawInferencePatchDataset requires an explicit non-empty patch_size")
        if not (0.0 <= float(overlap) < 1.0):
            _fail_validation(f"overlap must be in [0, 1), got {overlap}")
        self.images_dir = str(images_dir)
        self.config = config
        self.overlap = float(overlap)
        self.image_reader_name = str(image_reader)
        self.multi_image = bool(multi_image)
        self.transform = transform
        self.preprocessor = TaskAwarePreprocessor(config, verbose=False)

        grouped = _scan_image_dir(self.images_dir, self.multi_image)
        self.cases: list[RawInferenceCase] = []
        self.index_map: list[tuple[int, int]] = []

        for identifier, image_files in sorted(grouped.items()):
            reader = _build_reader(self.image_reader_name, image_files[0])
            case = self.preprocessor.run_task_case_from_files(
                image_files=image_files,
                image_reader=reader,
                task_mode=TaskMode.PAIRED_GENERATIVE,
                run_stage=RunStage.PREDICT,
            )
            resolved_patch_size = _resolve_patch_size(case.image, patch_size, "raw inference")
            patch_starts = _compute_patch_starts(case.image.shape[1:], resolved_patch_size, self.overlap)
            case_record = RawInferenceCase(
                identifier=identifier,
                image_files=list(image_files),
                image=np.asarray(case.image, dtype=np.float32),
                properties=case.properties,
                patch_size=resolved_patch_size,
                patch_starts=patch_starts,
            )
            case_index = len(self.cases)
            self.cases.append(case_record)
            self.index_map.extend((case_index, patch_index) for patch_index in range(len(patch_starts)))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case_index, patch_index = self.index_map[index]
        case = self.cases[case_index]
        starts = case.patch_starts[patch_index]
        image = _crop_with_starts(case.image, case.patch_size, starts)
        sample: Dict[str, Any] = {
            "image": torch.from_numpy(np.asarray(image)).float(),
            "identifier": case.identifier,
            "case_index": int(case_index),
            "patch_index": int(patch_index),
            "starts": torch.as_tensor(starts, dtype=torch.long),
            "patch_size": torch.as_tensor(case.patch_size, dtype=torch.long),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_case(self, case_index: int) -> RawInferenceCase:
        return self.cases[int(case_index)]

    def build_accumulator(self, case_index: int, channels: int = 1) -> InferencePatchAccumulator:
        case = self.get_case(case_index)
        return InferencePatchAccumulator(case.image.shape[1:], channels=channels)

    def restore_prediction(self, prediction: np.ndarray, case_index: int) -> np.ndarray:
        case = self.get_case(case_index)
        denormalized = _inverse_normalize_prediction(np.asarray(prediction), self.config)
        return _undo_preprocessing(denormalized, case.properties, self.config)

    def save_prediction_nifti(
        self,
        prediction: np.ndarray,
        case_index: int,
        output_path: str | Path,
    ) -> None:
        case = self.get_case(case_index)
        restored = self.restore_prediction(prediction, case_index)
        _save_nifti_like_reference(restored, case.image_files[0], output_path)
