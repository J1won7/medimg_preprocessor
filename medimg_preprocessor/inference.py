from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:
        pass

from .config import PreprocessingConfig, ResamplingConfig
from .dataset import load_preprocessed_dataset_manifest
from .geometry import resample_image, resample_mask
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
    if root.is_file():
        if multi_image:
            _fail_validation("multi_image=True requires a directory containing all channel files")
        if _detect_file_ending(root.name) not in SUPPORTED_SCAN_ENDINGS:
            _fail_validation(f"Unsupported image file: {folder}")
        return {_strip_known_suffix(root.name): [str(root)]}
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


def _config_from_manifest_payload(payload: Any) -> PreprocessingConfig:
    if not isinstance(payload, dict):
        _fail_validation("Manifest does not contain a preprocessing_config mapping")
    required_fields = (
        "spacing",
        "transpose_forward",
        "normalization_schemes",
        "use_mask_for_norm",
    )
    missing = [field for field in required_fields if field not in payload]
    if missing:
        _fail_validation(
            "Manifest preprocessing_config is missing required fields: " + ", ".join(missing)
        )
    resampling_payload = payload.get("resampling", {})
    if not isinstance(resampling_payload, dict):
        _fail_validation("Manifest preprocessing_config.resampling must be a mapping")
    try:
        resampling = ResamplingConfig.from_mapping(resampling_payload)
        return PreprocessingConfig(
            spacing=payload["spacing"],
            transpose_forward=payload["transpose_forward"],
            normalization_schemes=payload["normalization_schemes"],
            use_mask_for_norm=payload["use_mask_for_norm"],
            foreground_intensity_properties_per_channel=payload.get(
                "foreground_intensity_properties_per_channel", {}
            ),
            resampling=resampling,
        )
    except (TypeError, ValueError) as error:
        _fail_validation(f"Invalid preprocessing_config in manifest: {error}")


def _patch_size_from_manifest(manifest: dict, configuration: Optional[str]) -> Optional[tuple[int, ...]]:
    configurations = manifest.get("configurations") or {}
    selected_configuration = configuration or manifest.get("default_configuration")
    if selected_configuration is not None:
        if selected_configuration not in configurations:
            _fail_validation(
                f"Configuration '{selected_configuration}' was not found in the preprocessing manifest"
            )
        patch_size = configurations[selected_configuration].get("patch_size")
        if patch_size is not None:
            return tuple(int(value) for value in patch_size)
    elif len(configurations) == 1:
        only_configuration = next(iter(configurations.values()))
        patch_size = only_configuration.get("patch_size")
        if patch_size is not None:
            return tuple(int(value) for value in patch_size)

    patch_size = manifest.get("default_patch_size")
    if patch_size is None:
        return None
    return tuple(int(value) for value in patch_size)


def _manifest_config_and_task(
    manifest: dict,
    domain: Optional[str],
) -> tuple[PreprocessingConfig, str, Optional[str]]:
    task_mode = manifest["task_mode"]
    dataset_kind = manifest["dataset_kind"]
    if task_mode == TaskMode.SELF_SUPERVISED:
        _fail_validation("self_supervised manifests cannot be used for inference")
    if dataset_kind == "single_folder":
        if domain is not None:
            _fail_validation("domain is only valid for unpaired_generative manifests")
        return _config_from_manifest_payload(manifest.get("preprocessing_config")), task_mode, None
    if dataset_kind != "unpaired_domains":
        _fail_validation(f"Unsupported manifest dataset_kind '{dataset_kind}'")
    if domain is None:
        _fail_validation("unpaired_generative inference requires domain='a' or domain='b'")
    domain = str(domain).lower()
    if domain not in {"a", "b"}:
        _fail_validation(f"domain must be 'a' or 'b', got '{domain}'")
    domain_payload = manifest["domains"].get(domain)
    if not isinstance(domain_payload, dict):
        _fail_validation(f"Manifest does not contain domain '{domain}'")
    return _config_from_manifest_payload(domain_payload.get("preprocessing_config")), task_mode, domain


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


def _output_starts_for_padded_input(
    spatial_shape: Sequence[int],
    patch_size: Sequence[int],
    input_starts: Sequence[Sequence[int]],
) -> list[tuple[int, ...]]:
    padding_before = [max(int(patch) - int(size), 0) // 2 for size, patch in zip(spatial_shape, patch_size)]
    return [
        tuple(int(start) - int(before) for start, before in zip(starts, padding_before))
        for starts in input_starts
    ]


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


def _undo_preprocessing(
    pred: np.ndarray,
    properties: Dict[str, Any],
    config: PreprocessingConfig,
    *,
    resampling_role: str = "image",
    is_segmentation: bool = False,
) -> np.ndarray:
    restored = pred.astype(np.float32, copy=False)
    settings = properties.get("medimg_preprocessor_settings", {})

    if settings.get("resample", True):
        shape_before_resampling = tuple(int(i) for i in properties["shape_before_resampling"])
        orders = config.resampling.orders_for(resampling_role, restored.ndim - 1)
        current_spacing = properties.get(
            "spacing_after_resampling",
            config.spacing,
        )
        new_spacing = properties.get(
            "spacing_after_transpose",
            config.spacing,
        )
        resample = resample_mask if is_segmentation else resample_image
        restored = resample(
            restored,
            shape_before_resampling,
            orders=orders,
            current_spacing=current_spacing,
            new_spacing=new_spacing,
        )

    if settings.get("transpose", True):
        inverse_axes = np.argsort(np.asarray(config.transpose_forward))
        restored = restored.transpose((0, *[int(i) + 1 for i in inverse_axes]))

    return restored


def _save_nifti_like_reference(volume: np.ndarray, reference_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    try:
        import nibabel as nib
    except ModuleNotFoundError:
        _fail_validation("nibabel is required to save NIfTI inference outputs")
    reference = nib.load(str(reference_path))
    volume = np.asarray(volume)
    if volume.ndim == 4:
        data = volume.transpose((3, 2, 1, 0))
        if volume.shape[0] == 1:
            data = data[..., 0]
    elif volume.ndim == 3:
        data = volume.transpose((2, 1, 0))
        if volume.shape[0] == 1:
            data = data[..., 0]
    else:
        _fail_validation(f"NIfTI export expects a 2D or 3D channel-first volume, got {volume.shape}")
    data = data.astype(np.float32, copy=False)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data, affine=reference.affine, header=header)
    image.set_qform(reference.get_qform(), code=int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), code=int(reference.header["sform_code"]))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))


@dataclass
class RawInferenceCase:
    identifier: str
    image_files: list[str]
    image: np.ndarray
    properties: dict
    patch_size: tuple[int, ...]
    patch_starts: list[tuple[int, ...]]
    output_starts: list[tuple[int, ...]]


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
        if patch.ndim != len(self.spatial_shape) + 1:
            _fail_validation(
                f"patch must have {len(self.spatial_shape) + 1} dimensions including channels, got {patch.shape}"
            )
        if patch.shape[0] != self.channels:
            _fail_validation(f"patch has {patch.shape[0]} channels, accumulator expects {self.channels}")
        if len(starts) != len(self.spatial_shape):
            _fail_validation("starts must have the same dimensionality as the accumulator spatial shape")

        destination_slices = []
        source_slices = []
        for start, patch_length, spatial_length in zip(starts, patch.shape[1:], self.spatial_shape):
            start = int(start)
            source_start = max(0, -start)
            destination_start = max(0, start)
            length = min(int(patch_length) - source_start, int(spatial_length) - destination_start)
            if length <= 0:
                return
            source_slices.append(slice(source_start, source_start + length))
            destination_slices.append(slice(destination_start, destination_start + length))
        destination = (slice(None),) + tuple(destination_slices)
        source = (slice(None),) + tuple(source_slices)
        self.value_sum[destination] += patch[source]
        self.value_count[(slice(None),) + tuple(destination_slices)] += 1.0

    def finalize(self) -> np.ndarray:
        return self.value_sum / np.clip(self.value_count, 1e-8, None)


class RawInferencePatchDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        config: PreprocessingConfig,
        patch_size: Optional[Sequence[int]],
        *,
        overlap: float = 0.5,
        image_reader: str = "auto",
        multi_image: bool = False,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        task_mode: str = TaskMode.PAIRED_GENERATIVE,
    ):
        _require_torch()
        if patch_size is not None and len(tuple(patch_size)) == 0:
            _fail_validation("patch_size must be non-empty when provided")
        if not (0.0 <= float(overlap) < 1.0):
            _fail_validation(f"overlap must be in [0, 1), got {overlap}")
        if task_mode == TaskMode.SELF_SUPERVISED:
            _fail_validation("self_supervised does not support inference")
        self.images_dir = str(images_dir)
        self.config = config
        self.overlap = float(overlap)
        self.image_reader_name = str(image_reader)
        self.multi_image = bool(multi_image)
        self.transform = transform
        self.task_mode = str(task_mode)
        self.requested_patch_size = None if patch_size is None else tuple(int(value) for value in patch_size)
        self.preprocessor = TaskAwarePreprocessor(config, verbose=False)

        grouped = _scan_image_dir(self.images_dir, self.multi_image)
        self.cases: list[RawInferenceCase] = []
        self.index_map: list[tuple[int, int]] = []

        for identifier, image_files in sorted(grouped.items()):
            reader = _build_reader(self.image_reader_name, image_files[0])
            case = self.preprocessor.run_task_case_from_files(
                image_files=image_files,
                image_reader=reader,
                task_mode=self.task_mode,
                run_stage=RunStage.PREDICT,
            )
            resolved_patch_size = (
                tuple(int(value) for value in case.image.shape[1:])
                if self.requested_patch_size is None
                else _resolve_patch_size(case.image, self.requested_patch_size, "raw inference")
            )
            patch_starts = _compute_patch_starts(case.image.shape[1:], resolved_patch_size, self.overlap)
            output_starts = _output_starts_for_padded_input(
                case.image.shape[1:],
                resolved_patch_size,
                patch_starts,
            )
            case_record = RawInferenceCase(
                identifier=identifier,
                image_files=list(image_files),
                image=np.asarray(case.image, dtype=np.float32),
                properties=case.properties,
                patch_size=resolved_patch_size,
                patch_starts=patch_starts,
                output_starts=output_starts,
            )
            case_index = len(self.cases)
            self.cases.append(case_record)
            self.index_map.extend((case_index, patch_index) for patch_index in range(len(patch_starts)))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        case_index, patch_index = self.index_map[index]
        case = self.cases[case_index]
        input_starts = case.patch_starts[patch_index]
        starts = case.output_starts[patch_index]
        image = _crop_with_starts(case.image, case.patch_size, input_starts)
        sample: Dict[str, Any] = {
            "image": torch.from_numpy(np.asarray(image)).float(),
            "identifier": case.identifier,
            "case_index": int(case_index),
            "patch_index": int(patch_index),
            "starts": torch.as_tensor(starts, dtype=torch.long),
            "input_starts": torch.as_tensor(input_starts, dtype=torch.long),
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

    def build_accumulators(self, channels: int = 1) -> list[InferencePatchAccumulator]:
        return [self.build_accumulator(case_index, channels=channels) for case_index in range(len(self.cases))]

    def accumulate_batch(
        self,
        accumulators: Sequence[InferencePatchAccumulator],
        predictions: Any,
        case_indices: Sequence[int],
        starts: Sequence[Sequence[int]],
    ) -> None:
        if len(accumulators) != len(self.cases):
            _fail_validation(f"Expected {len(self.cases)} accumulators, got {len(accumulators)}")
        if torch is not None and isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        predictions = np.asarray(predictions)
        if predictions.ndim < 2:
            _fail_validation("predictions must have a batch dimension and at least one spatial dimension")
        if torch is not None and isinstance(case_indices, torch.Tensor):
            case_indices = case_indices.detach().cpu().numpy()
        if torch is not None and isinstance(starts, torch.Tensor):
            starts = starts.detach().cpu().numpy()
        if len(predictions) != len(case_indices) or len(predictions) != len(starts):
            _fail_validation("predictions, case_indices, and starts must have the same batch length")
        for prediction, case_index, patch_starts in zip(predictions, case_indices, starts):
            case_index = int(case_index)
            if case_index < 0 or case_index >= len(self.cases):
                _fail_validation(f"Invalid case_index {case_index}")
            accumulators[case_index].add_patch(prediction, patch_starts)

    def restore_prediction(
        self,
        prediction: np.ndarray,
        case_index: int,
        *,
        prediction_kind: Optional[str] = None,
    ) -> np.ndarray:
        case = self.get_case(case_index)
        if prediction_kind is None:
            prediction_kind = "label" if self.task_mode == TaskMode.SEGMENTATION else "image"
        prediction_kind = str(prediction_kind).lower()
        if prediction_kind == "image":
            restored = _inverse_normalize_prediction(np.asarray(prediction), self.config)
            return _undo_preprocessing(restored, case.properties, self.config)
        if prediction_kind == "label":
            return _undo_preprocessing(
                np.asarray(prediction),
                case.properties,
                self.config,
                resampling_role="label",
                is_segmentation=True,
            )
        _fail_validation("prediction_kind must be 'image' or 'label'")

    def save_prediction_nifti(
        self,
        prediction: np.ndarray,
        case_index: int,
        output_path: Union[str, Path],
        *,
        prediction_kind: Optional[str] = None,
    ) -> None:
        case = self.get_case(case_index)
        restored = self.restore_prediction(prediction, case_index, prediction_kind=prediction_kind)
        _save_nifti_like_reference(restored, case.image_files[0], output_path)


class ManifestInferencePatchDataset(RawInferencePatchDataset):
    """Run manifest-matched preprocessing and sliding-window patch inference on raw images."""

    def __init__(
        self,
        preprocessed_folder: str,
        images_dir: str,
        *,
        patch_size: Optional[Sequence[int]] = None,
        configuration: Optional[str] = None,
        domain: Optional[str] = None,
        overlap: float = 0.5,
        image_reader: Optional[str] = None,
        multi_image: Optional[bool] = None,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        manifest = load_preprocessed_dataset_manifest(str(preprocessed_folder))
        config, task_mode, selected_domain = _manifest_config_and_task(manifest, domain)
        resolved_patch_size = (
            tuple(int(value) for value in patch_size)
            if patch_size is not None
            else _patch_size_from_manifest(manifest, configuration)
        )
        self.preprocessed_folder = str(preprocessed_folder)
        self.manifest = manifest
        self.configuration = configuration or manifest.get("default_configuration")
        self.domain = selected_domain
        input_metadata = (
            manifest if selected_domain is None else manifest["domains"][selected_domain]
        )
        resolved_reader = image_reader if image_reader is not None else (input_metadata.get("image_reader") or "auto")
        resolved_multi_image = (
            bool(multi_image)
            if multi_image is not None
            else bool(input_metadata.get("multi_image", False))
        )
        super().__init__(
            images_dir=images_dir,
            config=config,
            patch_size=resolved_patch_size,
            overlap=overlap,
            image_reader=resolved_reader,
            multi_image=resolved_multi_image,
            transform=transform,
            task_mode=task_mode,
        )
