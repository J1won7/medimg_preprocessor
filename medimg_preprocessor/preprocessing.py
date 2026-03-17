from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np

from .config import PreprocessingConfig
from .geometry import compute_new_shape, create_nonzero_mask, crop_to_nonzero, resample_array
from .normalization import get_normalizer


class PreprocessingMode:
    SEGMENTATION = "segmentation"
    GENERATIVE = "generative"


class TaskMode:
    SEGMENTATION = "segmentation"
    PAIRED_GENERATIVE = "paired_generative"
    UNPAIRED_GENERATIVE = "unpaired_generative"
    SELF_SUPERVISED = "self_supervised"


class RunStage:
    TRAIN = "train"
    PREDICT = "predict"
    PREDICT_AND_EVALUATE = "predict_and_evaluate"


@dataclass
class ModularPreprocessingSettings:
    mode: str = PreprocessingMode.SEGMENTATION
    transpose: bool = True
    crop_to_nonzero: bool = True
    normalize: bool = True
    resample: bool = True
    keep_target: Optional[bool] = None
    keep_nonzero_mask: bool = False
    collect_foreground_locations: Optional[bool] = None
    use_nonzero_mask_for_norm_if_no_target: bool = True

    @classmethod
    def segmentation_defaults(cls) -> "ModularPreprocessingSettings":
        return cls(mode=PreprocessingMode.SEGMENTATION, keep_target=True, collect_foreground_locations=True)

    @classmethod
    def generative_defaults(cls) -> "ModularPreprocessingSettings":
        return cls(mode=PreprocessingMode.GENERATIVE, keep_target=False, collect_foreground_locations=False)


@dataclass
class TaskPreprocessedCase:
    image: np.ndarray
    properties: dict
    target: Optional[np.ndarray] = None
    target_properties: Optional[dict] = None
    evaluation_reference: Optional[np.ndarray] = None
    evaluation_properties: Optional[dict] = None
    task_mode: Optional[str] = None
    run_stage: Optional[str] = None
    reference_type: Optional[str] = None


def _fail_validation(message: str) -> None:
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)


def _resolve_settings(settings: Optional[ModularPreprocessingSettings], mode: str) -> ModularPreprocessingSettings:
    if settings is None:
        return (
            ModularPreprocessingSettings.segmentation_defaults()
            if mode == PreprocessingMode.SEGMENTATION
            else ModularPreprocessingSettings.generative_defaults()
        )
    settings = replace(settings)
    if settings.keep_target is None:
        settings.keep_target = mode == PreprocessingMode.SEGMENTATION
    if settings.collect_foreground_locations is None:
        settings.collect_foreground_locations = mode == PreprocessingMode.SEGMENTATION
    settings.mode = mode
    return settings


class ModularPreprocessor:
    def __init__(self, config: PreprocessingConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._validate_config()

    def _validate_config(self) -> None:
        num_channels = len(self.config.normalization_schemes)
        if len(self.config.use_mask_for_norm) != num_channels:
            _fail_validation(
                "normalization_schemes and use_mask_for_norm must have identical length, "
                f"got {len(self.config.normalization_schemes)} and {len(self.config.use_mask_for_norm)}"
            )

    def _validate_properties(self, properties: dict, spatial_dims: int, context: str) -> tuple[float, ...]:
        if not isinstance(properties, dict):
            _fail_validation(f"{context} properties must be a dict, got {type(properties).__name__}")
        if "spacing" not in properties:
            _fail_validation(f"{context} properties must contain a 'spacing' entry")
        spacing = tuple(float(i) for i in properties["spacing"])
        if len(spacing) != spatial_dims:
            _fail_validation(
                f"{context} spacing must have {spatial_dims} values, got {len(spacing)} for spacing {spacing}"
            )
        if any(not np.isfinite(i) for i in spacing):
            _fail_validation(f"{context} spacing must be finite, got {spacing}")
        if any(i <= 0 for i in spacing):
            _fail_validation(f"{context} spacing must be positive, got {spacing}")
        return spacing

    def _validate_array(self, array: np.ndarray, name: str) -> None:
        if not isinstance(array, np.ndarray):
            _fail_validation(f"{name} must be a numpy.ndarray, got {type(array).__name__}")
        if array.ndim not in (3, 4):
            _fail_validation(
                f"{name} must have shape (C, X, Y) or (C, X, Y, Z), got array with shape {array.shape}"
            )
        if array.shape[0] <= 0:
            _fail_validation(f"{name} must have at least one channel, got shape {array.shape}")
        if any(i <= 0 for i in array.shape[1:]):
            _fail_validation(f"{name} spatial dimensions must be non-empty, got shape {array.shape}")
        if not np.issubdtype(array.dtype, np.number):
            _fail_validation(f"{name} must contain numeric data, got dtype {array.dtype}")
        if not np.all(np.isfinite(array)):
            _fail_validation(f"{name} must contain only finite numeric values, got non-finite values")

    def _validate_target(
        self,
        image: np.ndarray,
        target: np.ndarray,
        target_is_segmentation: bool,
        name: str,
    ) -> None:
        self._validate_array(target, name)
        if target.ndim != image.ndim:
            _fail_validation(
                f"{name} must have the same number of dimensions as image, got {target.shape} and {image.shape}"
            )
        if tuple(target.shape[1:]) != tuple(image.shape[1:]):
            _fail_validation(
                f"{name} spatial shape must match image, got {target.shape[1:]} and {image.shape[1:]}"
            )
        if target_is_segmentation and target.shape[0] != 1:
            _fail_validation(f"Segmentation target must have exactly one channel, got shape {target.shape}")

    def _validate_case_inputs(
        self,
        image: np.ndarray,
        properties: dict,
        target: np.ndarray | None,
        target_is_segmentation: bool,
    ) -> tuple[float, ...]:
        self._validate_array(image, "image")
        spacing = self._validate_properties(properties, image.ndim - 1, "image")
        if image.ndim - 1 != len(self.config.spacing):
            _fail_validation(
                f"Config spacing has {len(self.config.spacing)} dims but image has spatial dims {image.shape[1:]}"
            )
        if image.shape[0] != len(self.config.normalization_schemes):
            _fail_validation(
                f"Image channel count {image.shape[0]} does not match normalization config length "
                f"{len(self.config.normalization_schemes)}"
            )
        if target is not None:
            self._validate_target(image, target, target_is_segmentation, "target")
        return spacing

    def _normalize(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        intensity_properties_per_channel: Optional[dict],
    ) -> np.ndarray:
        stats = intensity_properties_per_channel or self.config.foreground_intensity_properties_per_channel
        for c in range(image.shape[0]):
            scheme = self.config.normalization_schemes[c]
            normalizer_cls = get_normalizer(scheme)
            channel_stats = stats.get(str(c), {})
            if normalizer_cls.requires_intensity_properties:
                required = {"percentile_00_5", "percentile_99_5", "mean", "std"}
                missing = sorted(required.difference(channel_stats.keys()))
                if missing:
                    _fail_validation(
                        f"Normalization scheme '{scheme}' for channel {c} requires intensity statistics "
                        f"{missing}, but they were not provided."
                    )
            normalizer = normalizer_cls(
                use_mask_for_norm=self.config.use_mask_for_norm[c],
                intensity_properties=channel_stats,
            )
            image[c] = normalizer.run(image[c], mask)
        return image

    @staticmethod
    def _sample_foreground_locations(
        target: np.ndarray,
        labels: Sequence[int] | None = None,
        seed: int = 1234,
        max_samples: int = 10000,
    ) -> dict:
        if labels is None:
            labels = [int(i) for i in np.unique(target) if i > 0]
        rng = np.random.RandomState(seed)
        locations = {}
        for label in labels:
            coords = np.argwhere(target[0] == label)
            if len(coords) == 0:
                locations[int(label)] = []
                continue
            picked = coords[rng.choice(len(coords), min(max_samples, len(coords)), replace=False)]
            locations[int(label)] = np.concatenate(
                [np.zeros((picked.shape[0], 1), dtype=np.int64), picked.astype(np.int64)], axis=1
            )
        return locations

    def run_case(
        self,
        image: np.ndarray,
        properties: dict,
        target: np.ndarray | None = None,
        settings: Optional[ModularPreprocessingSettings] = None,
        intensity_properties_per_channel: Optional[dict] = None,
        target_is_segmentation: bool = True,
    ) -> tuple[np.ndarray, Optional[np.ndarray], dict]:
        mode = PreprocessingMode.SEGMENTATION if target_is_segmentation else PreprocessingMode.GENERATIVE
        settings = _resolve_settings(settings, mode)
        properties = dict(properties)
        properties["medimg_preprocessor_settings"] = asdict(settings)

        image = image.astype(np.float32, copy=True)
        target = None if target is None else np.copy(target)
        spacing = list(self._validate_case_inputs(image, properties, target, target_is_segmentation))
        if settings.transpose:
            axes = [0, *[i + 1 for i in self.config.transpose_forward]]
            image = image.transpose(axes)
            if target is not None:
                target = target.transpose(axes)
            spacing = [spacing[i] for i in self.config.transpose_forward]
        properties["spacing_after_transpose"] = spacing
        properties["shape_before_cropping"] = tuple(int(i) for i in image.shape[1:])

        mask = None
        if settings.crop_to_nonzero:
            image, cropped_reference, bbox = crop_to_nonzero(image, target)
            properties["bbox_used_for_cropping"] = bbox
            if target is None:
                mask = cropped_reference[0] >= 0
            else:
                target = cropped_reference
                mask = target[0] >= 0
        else:
            properties["bbox_used_for_cropping"] = None
            if settings.keep_nonzero_mask or (settings.normalize and any(self.config.use_mask_for_norm)):
                mask = create_nonzero_mask(image)
        properties["shape_after_cropping_and_before_resampling"] = image.shape[1:]

        if settings.normalize:
            if mask is None and settings.use_nonzero_mask_for_norm_if_no_target and any(self.config.use_mask_for_norm):
                mask = create_nonzero_mask(image)
            image = self._normalize(image, mask, intensity_properties_per_channel)

        if settings.resample:
            target_spacing = list(self.config.spacing)
            if len(target_spacing) < len(image.shape[1:]):
                target_spacing = [spacing[0]] + target_spacing
            new_shape = compute_new_shape(image.shape[1:], spacing, target_spacing)
            image = resample_array(
                image,
                new_shape,
                spacing,
                target_spacing,
                is_seg=False,
                order=self.config.resampling.image_order,
                order_z=self.config.resampling.image_order_z,
                force_separate_z=self.config.resampling.force_separate_z,
                separate_z_anisotropy_threshold=self.config.resampling.separate_z_anisotropy_threshold,
            )
            if target is not None:
                if target_is_segmentation:
                    target = resample_array(
                        target,
                        new_shape,
                        spacing,
                        target_spacing,
                        is_seg=True,
                        order=self.config.resampling.label_order,
                        order_z=self.config.resampling.label_order_z,
                        force_separate_z=self.config.resampling.force_separate_z,
                        separate_z_anisotropy_threshold=self.config.resampling.separate_z_anisotropy_threshold,
                    )
                else:
                    target = resample_array(
                        target,
                        new_shape,
                        spacing,
                        target_spacing,
                        is_seg=False,
                        order=self.config.resampling.image_order,
                        order_z=self.config.resampling.image_order_z,
                        force_separate_z=self.config.resampling.force_separate_z,
                        separate_z_anisotropy_threshold=self.config.resampling.separate_z_anisotropy_threshold,
                    )
            properties["spacing_after_resampling"] = target_spacing
            properties["shape_after_resampling"] = tuple(int(i) for i in new_shape)
        else:
            properties["spacing_after_resampling"] = spacing
            properties["shape_after_resampling"] = image.shape[1:]

        if target is not None and settings.collect_foreground_locations and target_is_segmentation:
            properties["class_locations"] = self._sample_foreground_locations(target)

        if not settings.keep_target:
            target = None
        if target is not None and target_is_segmentation:
            target = target.astype(np.int16 if (np.max(target) > 127 or np.min(target) < -128) else np.int8, copy=False)
        return image, target, properties


class SegmentationPreprocessor(ModularPreprocessor):
    def run_case(self, image: np.ndarray, properties: dict, target: np.ndarray | None = None, **kwargs):
        settings = kwargs.pop("settings", None) or ModularPreprocessingSettings.segmentation_defaults()
        return super().run_case(image, properties, target=target, settings=settings, target_is_segmentation=True, **kwargs)


class GenerativePreprocessor(ModularPreprocessor):
    def run_case(self, image: np.ndarray, properties: dict, target: np.ndarray | None = None, **kwargs):
        settings = kwargs.pop("settings", None) or ModularPreprocessingSettings.generative_defaults()
        return super().run_case(image, properties, target=target, settings=settings, target_is_segmentation=False, **kwargs)


class TaskAwarePreprocessor:
    def __init__(self, config: PreprocessingConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self.segmentation_preprocessor = SegmentationPreprocessor(config, verbose=verbose)
        self.generative_preprocessor = GenerativePreprocessor(config, verbose=verbose)

    def _validate_task_request(
        self,
        task_mode: str,
        run_stage: str,
        image: np.ndarray,
        image_properties: dict,
        reference: Optional[np.ndarray],
        reference_properties: Optional[dict],
    ) -> None:
        valid_stages = {
            TaskMode.SEGMENTATION: {RunStage.TRAIN, RunStage.PREDICT, RunStage.PREDICT_AND_EVALUATE},
            TaskMode.PAIRED_GENERATIVE: {RunStage.TRAIN, RunStage.PREDICT, RunStage.PREDICT_AND_EVALUATE},
            TaskMode.UNPAIRED_GENERATIVE: {RunStage.TRAIN, RunStage.PREDICT},
            TaskMode.SELF_SUPERVISED: {RunStage.TRAIN},
        }
        if task_mode not in valid_stages:
            _fail_validation(f"Unsupported task_mode '{task_mode}'")
        if run_stage not in valid_stages[task_mode]:
            _fail_validation(f"Unsupported run_stage '{run_stage}' for task_mode '{task_mode}'")

        self.generative_preprocessor._validate_array(image, "image")
        self.generative_preprocessor._validate_properties(image_properties, image.ndim - 1, "image")

        if task_mode == TaskMode.SEGMENTATION and run_stage == RunStage.TRAIN:
            if reference is None:
                _fail_validation("segmentation train requires a segmentation reference")
            self.segmentation_preprocessor._validate_target(image, reference, True, "segmentation reference")
            if reference_properties is not None:
                self.generative_preprocessor._validate_properties(
                    reference_properties,
                    reference.ndim - 1,
                    "segmentation reference",
                )

        if task_mode == TaskMode.SEGMENTATION and run_stage == RunStage.PREDICT_AND_EVALUATE:
            if reference is None:
                _fail_validation("segmentation predict_and_evaluate requires an evaluation reference")
            self.segmentation_preprocessor._validate_target(image, reference, True, "evaluation reference")
            if reference_properties is not None:
                self.generative_preprocessor._validate_properties(
                    reference_properties,
                    reference.ndim - 1,
                    "evaluation reference",
                )

        if task_mode == TaskMode.SEGMENTATION and run_stage == RunStage.PREDICT and reference is not None:
            _fail_validation("segmentation predict does not accept a reference; use predict_and_evaluate instead")

        if task_mode == TaskMode.PAIRED_GENERATIVE and run_stage == RunStage.TRAIN:
            if reference is None:
                _fail_validation("paired_generative train requires a paired reference image")
            self.generative_preprocessor._validate_target(image, reference, False, "paired reference")
            if reference_properties is None:
                _fail_validation("paired_generative train requires reference_properties")
            self.generative_preprocessor._validate_properties(
                reference_properties,
                reference.ndim - 1,
                "paired reference",
            )

        if task_mode == TaskMode.PAIRED_GENERATIVE and run_stage == RunStage.PREDICT_AND_EVALUATE:
            if reference is None:
                _fail_validation("paired_generative predict_and_evaluate requires an evaluation reference image")
            self.generative_preprocessor._validate_target(image, reference, False, "evaluation reference")
            if reference_properties is None:
                _fail_validation("paired_generative predict_and_evaluate requires reference_properties")
            self.generative_preprocessor._validate_properties(
                reference_properties,
                reference.ndim - 1,
                "evaluation reference",
            )

        if task_mode == TaskMode.PAIRED_GENERATIVE and run_stage == RunStage.PREDICT and reference is not None:
            _fail_validation("paired_generative predict does not accept a reference; use predict_and_evaluate instead")

        if task_mode == TaskMode.UNPAIRED_GENERATIVE and reference is not None:
            _fail_validation("unpaired_generative does not accept a reference array")

        if task_mode == TaskMode.SELF_SUPERVISED and reference is not None:
            _fail_validation("self_supervised does not accept a reference array")

    def _run_paired_image_case(
        self,
        image: np.ndarray,
        image_properties: dict,
        reference: np.ndarray,
        reference_properties: dict,
        settings: Optional[ModularPreprocessingSettings],
        input_intensity_properties_per_channel: Optional[dict],
        reference_intensity_properties_per_channel: Optional[dict],
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        settings = _resolve_settings(settings, PreprocessingMode.GENERATIVE)
        self.generative_preprocessor._validate_array(image, "source image")
        self.generative_preprocessor._validate_array(reference, "reference image")
        if image.ndim != reference.ndim:
            _fail_validation(
                f"paired_generative requires source and reference with matching ndim, got {image.shape} and {reference.shape}"
            )
        if tuple(image.shape[1:]) != tuple(reference.shape[1:]):
            _fail_validation(
                "paired_generative requires source and reference with matching spatial shape, "
                f"got {image.shape[1:]} and {reference.shape[1:]}"
            )
        image = image.astype(np.float32, copy=True)
        reference = reference.astype(np.float32, copy=True)
        properties = dict(image_properties)
        spacing = list(self.generative_preprocessor._validate_properties(image_properties, image.ndim - 1, "source image"))
        reference_spacing = self.generative_preprocessor._validate_properties(
            reference_properties,
            reference.ndim - 1,
            "reference image",
        )
        properties["medimg_preprocessor_settings"] = asdict(settings)
        properties["shape_before_cropping"] = tuple(int(i) for i in image.shape[1:])

        if tuple(spacing) != tuple(reference_spacing):
            _fail_validation("paired_generative expects source and target images with matching spacing")
        if image.shape[0] != len(self.config.normalization_schemes):
            _fail_validation(
                f"Source image channel count {image.shape[0]} does not match normalization config length "
                f"{len(self.config.normalization_schemes)}"
            )
        if reference.shape[0] != len(self.config.normalization_schemes):
            _fail_validation(
                f"Reference image channel count {reference.shape[0]} does not match normalization config length "
                f"{len(self.config.normalization_schemes)}"
            )

        if settings.transpose:
            axes = [0, *[i + 1 for i in self.config.transpose_forward]]
            image = image.transpose(axes)
            reference = reference.transpose(axes)
            spacing = [spacing[i] for i in self.config.transpose_forward]
        properties["spacing_after_transpose"] = spacing

        mask = None
        if settings.crop_to_nonzero:
            stacked, mask_ref, bbox = crop_to_nonzero(np.vstack((image, reference)), None)
            image = stacked[: image.shape[0]]
            reference = stacked[image.shape[0] :]
            mask = mask_ref[0] >= 0
            properties["bbox_used_for_cropping"] = bbox
        else:
            properties["bbox_used_for_cropping"] = None
            if settings.normalize and any(self.config.use_mask_for_norm):
                mask = create_nonzero_mask(np.vstack((image, reference)))
        properties["shape_after_cropping_and_before_resampling"] = image.shape[1:]

        if settings.normalize:
            image = self.generative_preprocessor._normalize(
                image,
                mask,
                input_intensity_properties_per_channel,
            )
            reference = self.generative_preprocessor._normalize(
                reference,
                mask,
                reference_intensity_properties_per_channel or input_intensity_properties_per_channel,
            )

        if settings.resample:
            target_spacing = list(self.config.spacing)
            if len(target_spacing) < len(image.shape[1:]):
                target_spacing = [spacing[0]] + target_spacing
            new_shape = compute_new_shape(image.shape[1:], spacing, target_spacing)
            image = resample_array(
                image,
                new_shape,
                spacing,
                target_spacing,
                is_seg=False,
                order=self.config.resampling.image_order,
                order_z=self.config.resampling.image_order_z,
                force_separate_z=self.config.resampling.force_separate_z,
                separate_z_anisotropy_threshold=self.config.resampling.separate_z_anisotropy_threshold,
            )
            reference = resample_array(
                reference,
                new_shape,
                spacing,
                target_spacing,
                is_seg=False,
                order=self.config.resampling.image_order,
                order_z=self.config.resampling.image_order_z,
                force_separate_z=self.config.resampling.force_separate_z,
                separate_z_anisotropy_threshold=self.config.resampling.separate_z_anisotropy_threshold,
            )
            properties["spacing_after_resampling"] = target_spacing
            properties["shape_after_resampling"] = tuple(int(i) for i in new_shape)
        else:
            properties["spacing_after_resampling"] = spacing
            properties["shape_after_resampling"] = image.shape[1:]
        return image, reference, properties

    def run_task_case(
        self,
        image: np.ndarray,
        image_properties: dict,
        task_mode: str,
        run_stage: str,
        reference: Optional[np.ndarray] = None,
        reference_properties: Optional[dict] = None,
        image_settings: Optional[ModularPreprocessingSettings] = None,
        input_intensity_properties_per_channel: Optional[dict] = None,
        reference_intensity_properties_per_channel: Optional[dict] = None,
    ) -> TaskPreprocessedCase:
        self._validate_task_request(task_mode, run_stage, image, image_properties, reference, reference_properties)
        if task_mode == TaskMode.SEGMENTATION and run_stage == RunStage.TRAIN:
            image_pp, target_pp, properties = self.segmentation_preprocessor.run_case(
                image,
                image_properties,
                target=reference,
                settings=image_settings,
                intensity_properties_per_channel=input_intensity_properties_per_channel,
            )
            return TaskPreprocessedCase(
                image=image_pp,
                properties=properties,
                target=target_pp,
                target_properties=dict(properties),
                task_mode=task_mode,
                run_stage=run_stage,
                reference_type="segmentation",
            )

        if task_mode == TaskMode.SEGMENTATION and run_stage == RunStage.PREDICT:
            image_pp, _, properties = self.generative_preprocessor.run_case(
                image,
                image_properties,
                target=None,
                settings=image_settings,
                intensity_properties_per_channel=input_intensity_properties_per_channel,
            )
            return TaskPreprocessedCase(
                image=image_pp,
                properties=properties,
                task_mode=task_mode,
                run_stage=run_stage,
                reference_type="none",
            )

        if task_mode == TaskMode.SEGMENTATION and run_stage == RunStage.PREDICT_AND_EVALUATE:
            image_pp, _, properties = self.generative_preprocessor.run_case(
                image,
                image_properties,
                target=None,
                settings=image_settings,
                intensity_properties_per_channel=input_intensity_properties_per_channel,
            )
            return TaskPreprocessedCase(
                image=image_pp,
                properties=properties,
                evaluation_reference=reference,
                evaluation_properties=reference_properties,
                task_mode=task_mode,
                run_stage=run_stage,
                reference_type="segmentation",
            )

        if task_mode == TaskMode.PAIRED_GENERATIVE and run_stage == RunStage.TRAIN:
            image_pp, target_pp, properties = self._run_paired_image_case(
                image,
                image_properties,
                reference,
                reference_properties or image_properties,
                image_settings,
                input_intensity_properties_per_channel,
                reference_intensity_properties_per_channel,
            )
            return TaskPreprocessedCase(
                image=image_pp,
                properties=properties,
                target=target_pp,
                target_properties=dict(properties),
                task_mode=task_mode,
                run_stage=run_stage,
                reference_type="image",
            )

        image_pp, _, properties = self.generative_preprocessor.run_case(
            image,
            image_properties,
            target=None,
            settings=image_settings,
            intensity_properties_per_channel=input_intensity_properties_per_channel,
        )
        case = TaskPreprocessedCase(
            image=image_pp,
            properties=properties,
            task_mode=task_mode,
            run_stage=run_stage,
            reference_type="none",
        )
        if run_stage == RunStage.PREDICT_AND_EVALUATE:
            case.evaluation_reference = reference
            case.evaluation_properties = reference_properties
        return case

    def run_task_case_from_files(
        self,
        image_files: Sequence[str],
        image_reader,
        task_mode: str,
        run_stage: str,
        reference_files: Optional[Sequence[str] | str] = None,
        reference_reader=None,
        image_settings: Optional[ModularPreprocessingSettings] = None,
        input_intensity_properties_per_channel: Optional[dict] = None,
        reference_intensity_properties_per_channel: Optional[dict] = None,
    ) -> TaskPreprocessedCase:
        if not hasattr(image_reader, "read_images"):
            _fail_validation("image_reader must provide a read_images(image_fnames) method")
        image, image_properties = image_reader.read_images(tuple(image_files))

        reference = None
        reference_properties = None
        if reference_files is not None:
            reader = reference_reader or image_reader
            if isinstance(reference_files, str):
                if not hasattr(reader, "read_seg"):
                    _fail_validation("reference_reader must provide read_seg(seg_fname) for string reference_files")
                reference, reference_properties = reader.read_seg(reference_files)
            else:
                if not hasattr(reader, "read_images"):
                    _fail_validation("reference_reader must provide read_images(image_fnames) for sequence reference_files")
                reference, reference_properties = reader.read_images(tuple(reference_files))

        return self.run_task_case(
            image=image,
            image_properties=image_properties,
            task_mode=task_mode,
            run_stage=run_stage,
            reference=reference,
            reference_properties=reference_properties,
            image_settings=image_settings,
            input_intensity_properties_per_channel=input_intensity_properties_per_channel,
            reference_intensity_properties_per_channel=reference_intensity_properties_per_channel,
        )

    def run_unpaired_case_pair(
        self,
        domain_a_image: np.ndarray,
        domain_a_properties: dict,
        domain_b_image: np.ndarray,
        domain_b_properties: dict,
        domain_a_settings: Optional[ModularPreprocessingSettings] = None,
        domain_b_settings: Optional[ModularPreprocessingSettings] = None,
        domain_a_intensity_properties_per_channel: Optional[dict] = None,
        domain_b_intensity_properties_per_channel: Optional[dict] = None,
    ) -> tuple[TaskPreprocessedCase, TaskPreprocessedCase]:
        case_a = self.run_task_case(
            domain_a_image,
            domain_a_properties,
            task_mode=TaskMode.UNPAIRED_GENERATIVE,
            run_stage=RunStage.TRAIN,
            image_settings=domain_a_settings,
            input_intensity_properties_per_channel=domain_a_intensity_properties_per_channel,
        )
        case_b = self.run_task_case(
            domain_b_image,
            domain_b_properties,
            task_mode=TaskMode.UNPAIRED_GENERATIVE,
            run_stage=RunStage.TRAIN,
            image_settings=domain_b_settings,
            input_intensity_properties_per_channel=domain_b_intensity_properties_per_channel,
        )
        return case_a, case_b


def compute_intensity_properties_from_image(
    image: np.ndarray,
    num_samples: int = 10000,
    seed: int = 1234,
    use_nonzero_mask: bool = True,
) -> dict:
    if not isinstance(image, np.ndarray):
        _fail_validation(f"image must be a numpy.ndarray, got {type(image).__name__}")
    if image.ndim not in (3, 4):
        _fail_validation(f"image must have shape (C, X, Y) or (C, X, Y, Z), got {image.shape}")
    rng = np.random.RandomState(seed)
    mask = create_nonzero_mask(image) if use_nonzero_mask else np.ones(image.shape[1:], dtype=bool)
    if not np.any(mask):
        mask = np.ones(image.shape[1:], dtype=bool)
    percentiles = np.array((0.5, 50.0, 99.5))
    stats = {}
    for c in range(image.shape[0]):
        values = image[c][mask]
        if values.size == 0:
            values = image[c].ravel()
        sampled = rng.choice(values, min(num_samples, values.size), replace=False)
        p005, median, p995 = np.percentile(sampled, percentiles)
        stats[str(c)] = {
            "mean": float(sampled.mean()),
            "median": float(median),
            "std": float(sampled.std()),
            "min": float(sampled.min()),
            "max": float(sampled.max()),
            "percentile_99_5": float(p995),
            "percentile_00_5": float(p005),
        }
    return stats


def aggregate_intensity_properties_from_arrays(
    images: Sequence[np.ndarray],
    num_samples_per_case: int = 10000,
    seed: int = 1234,
    use_nonzero_mask: bool = True,
) -> dict:
    if len(images) == 0:
        _fail_validation("aggregate_intensity_properties_from_arrays requires at least one image")
    per_channel: Dict[int, List[np.ndarray]] = {}
    rng = np.random.RandomState(seed)
    for image in images:
        if not isinstance(image, np.ndarray):
            _fail_validation(f"Each image must be a numpy.ndarray, got {type(image).__name__}")
        if image.ndim not in (3, 4):
            _fail_validation(f"Each image must have shape (C, X, Y) or (C, X, Y, Z), got {image.shape}")
        mask = create_nonzero_mask(image) if use_nonzero_mask else np.ones(image.shape[1:], dtype=bool)
        if not np.any(mask):
            mask = np.ones(image.shape[1:], dtype=bool)
        for c in range(image.shape[0]):
            values = image[c][mask]
            if values.size == 0:
                values = image[c].ravel()
            per_channel.setdefault(c, []).append(
                rng.choice(values, min(num_samples_per_case, values.size), replace=False)
            )
    result = {}
    percentiles = np.array((0.5, 50.0, 99.5))
    for c, chunks in per_channel.items():
        values = np.concatenate(chunks)
        p005, median, p995 = np.percentile(values, percentiles)
        result[str(c)] = {
            "mean": float(values.mean()),
            "median": float(median),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "percentile_99_5": float(p995),
            "percentile_00_5": float(p005),
        }
    return result


def aggregate_intensity_properties_from_image_files(
    image_files_per_case: Sequence[Sequence[str]],
    reader_fn: Callable[[Sequence[str]], tuple[np.ndarray, dict]],
    num_samples_per_case: int = 10000,
    seed: int = 1234,
    use_nonzero_mask: bool = True,
) -> dict:
    if not callable(reader_fn):
        _fail_validation("reader_fn must be callable")
    if len(image_files_per_case) == 0:
        _fail_validation("aggregate_intensity_properties_from_image_files requires at least one case")
    images = [reader_fn(files)[0] for files in image_files_per_case]
    return aggregate_intensity_properties_from_arrays(
        images,
        num_samples_per_case=num_samples_per_case,
        seed=seed,
        use_nonzero_mask=use_nonzero_mask,
    )
