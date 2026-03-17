from __future__ import annotations

import json
from dataclasses import dataclass, field
import math
from statistics import median
import warnings
from typing import Dict, List, Optional, Sequence, Union


@dataclass
class ResamplingConfig:
    image_order: int = 3
    image_order_z: int = 0
    label_order: int = 0
    label_order_z: int = 0
    force_separate_z: Optional[bool] = None
    separate_z_anisotropy_threshold: float = 3.0


@dataclass
class PreprocessingConfig:
    spacing: Sequence[float]
    transpose_forward: Sequence[int]
    normalization_schemes: Sequence[str]
    use_mask_for_norm: Sequence[bool]
    foreground_intensity_properties_per_channel: Dict[str, dict] = field(default_factory=dict)
    resampling: ResamplingConfig = field(default_factory=ResamplingConfig)

    def __post_init__(self) -> None:
        spacing = tuple(float(i) for i in self.spacing)
        transpose_forward = tuple(int(i) for i in self.transpose_forward)
        normalization_schemes = tuple(str(i) for i in self.normalization_schemes)
        use_mask_for_norm = tuple(bool(i) for i in self.use_mask_for_norm)

        if len(spacing) == 0:
            _fail_validation("PreprocessingConfig.spacing must contain at least one spatial dimension")
        if any(not math.isfinite(i) for i in spacing):
            _fail_validation(f"PreprocessingConfig.spacing must be finite, got {spacing}")
        if any(i <= 0 for i in spacing):
            _fail_validation(f"PreprocessingConfig.spacing must be positive, got {spacing}")
        if len(transpose_forward) != len(spacing):
            _fail_validation(
                "PreprocessingConfig.transpose_forward must have the same length as spacing, "
                f"got {len(transpose_forward)} and {len(spacing)}"
            )
        if sorted(transpose_forward) != list(range(len(spacing))):
            _fail_validation(
                "PreprocessingConfig.transpose_forward must be a permutation of spatial axes "
                f"0..{len(spacing) - 1}, got {transpose_forward}"
            )
        if len(normalization_schemes) == 0:
            _fail_validation("PreprocessingConfig.normalization_schemes must contain at least one channel scheme")
        if len(use_mask_for_norm) != len(normalization_schemes):
            _fail_validation(
                "PreprocessingConfig.use_mask_for_norm must match normalization_schemes length, "
                f"got {len(use_mask_for_norm)} and {len(normalization_schemes)}"
            )
        if self.resampling.image_order < 0 or self.resampling.label_order < 0:
            _fail_validation(
                f"Resampling orders must be non-negative, got image={self.resampling.image_order}, "
                f"label={self.resampling.label_order}"
            )
        if self.resampling.image_order_z < 0 or self.resampling.label_order_z < 0:
            _fail_validation(
                f"Z resampling orders must be non-negative, got image_z={self.resampling.image_order_z}, "
                f"label_z={self.resampling.label_order_z}"
            )
        if self.resampling.separate_z_anisotropy_threshold <= 0:
            _fail_validation(
                "ResamplingConfig.separate_z_anisotropy_threshold must be positive, "
                f"got {self.resampling.separate_z_anisotropy_threshold}"
            )

        self.spacing = spacing
        self.transpose_forward = transpose_forward
        self.normalization_schemes = normalization_schemes
        self.use_mask_for_norm = use_mask_for_norm

    @classmethod
    def from_nnunet_plans(
        cls,
        plans_or_file: Union[str, dict],
        configuration_name: str,
    ) -> "PreprocessingConfig":
        plans = plans_or_file
        if isinstance(plans_or_file, str):
            with open(plans_or_file, "r", encoding="utf-8") as f:
                plans = json.load(f)

        if "configurations" not in plans:
            _fail_validation("nnU-Net plans must contain a 'configurations' section")
        if "transpose_forward" not in plans:
            _fail_validation("nnU-Net plans must contain 'transpose_forward'")
        if configuration_name not in plans["configurations"]:
            _fail_validation(f"Configuration '{configuration_name}' was not found in nnU-Net plans")

        configuration = plans["configurations"][configuration_name]
        if "inherits_from" in configuration:
            configuration = _resolve_configuration_inheritance(plans, configuration_name)

        data_kwargs = configuration.get("resampling_fn_data_kwargs", {})
        seg_kwargs = configuration.get("resampling_fn_seg_kwargs", {})
        return cls(
            spacing=configuration["spacing"],
            transpose_forward=plans["transpose_forward"],
            normalization_schemes=configuration["normalization_schemes"],
            use_mask_for_norm=configuration["use_mask_for_norm"],
            foreground_intensity_properties_per_channel=plans.get(
                "foreground_intensity_properties_per_channel", {}
            ),
            resampling=ResamplingConfig(
                image_order=int(data_kwargs.get("order", 3)),
                image_order_z=int(data_kwargs.get("order_z", 0)),
                label_order=int(seg_kwargs.get("order", 0)),
                label_order_z=int(seg_kwargs.get("order_z", 0)),
                force_separate_z=data_kwargs.get("force_separate_z", None),
            ),
        )

    @classmethod
    def infer_from_dataset(
        cls,
        spacings: Sequence[Sequence[float]],
        num_channels: int,
        *,
        normalization_schemes: Optional[Sequence[str]] = None,
        use_mask_for_norm: Optional[Sequence[bool]] = None,
        foreground_intensity_properties_per_channel: Optional[Dict[str, dict]] = None,
        transpose_forward: Optional[Sequence[int]] = None,
        resampling: Optional[ResamplingConfig] = None,
    ) -> "PreprocessingConfig":
        if len(spacings) == 0:
            _fail_validation("infer_from_dataset requires at least one spacing entry")
        if num_channels <= 0:
            _fail_validation(f"infer_from_dataset requires a positive num_channels, got {num_channels}")

        normalized_spacings = [tuple(float(i) for i in spacing) for spacing in spacings]
        dims = len(normalized_spacings[0])
        if dims == 0:
            _fail_validation("spacing entries must contain at least one dimension")
        for spacing in normalized_spacings:
            if len(spacing) != dims:
                _fail_validation(
                    f"All spacing entries must have the same dimensionality, got {dims} and {len(spacing)}"
                )
            if any(not math.isfinite(i) or i <= 0 for i in spacing):
                _fail_validation(f"All spacing values must be finite and positive, got {spacing}")

        target_spacing = tuple(median(spacing[dim] for spacing in normalized_spacings) for dim in range(dims))
        if transpose_forward is None:
            transpose_forward = tuple(range(dims))
        if normalization_schemes is None:
            normalization_schemes = tuple("zscore" for _ in range(num_channels))
        if use_mask_for_norm is None:
            use_mask_for_norm = tuple(False for _ in range(num_channels))
        if foreground_intensity_properties_per_channel is None:
            foreground_intensity_properties_per_channel = {}
        if resampling is None:
            resampling = ResamplingConfig()

        return cls(
            spacing=target_spacing,
            transpose_forward=transpose_forward,
            normalization_schemes=normalization_schemes,
            use_mask_for_norm=use_mask_for_norm,
            foreground_intensity_properties_per_channel=foreground_intensity_properties_per_channel,
            resampling=resampling,
        )


def _resolve_configuration_inheritance(plans: dict, configuration_name: str) -> dict:
    configuration = dict(plans["configurations"][configuration_name])
    if "inherits_from" not in configuration:
        return configuration
    parent = _resolve_configuration_inheritance(plans, configuration["inherits_from"])
    parent.update(configuration)
    parent.pop("inherits_from", None)
    return parent


def _fail_validation(message: str) -> None:
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)
