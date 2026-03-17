from __future__ import annotations

import json
from dataclasses import dataclass, field
import math
import warnings
from typing import Dict, List, Optional, Sequence, Union


@dataclass
class ResamplingConfig:
    image_order: int = 3
    label_order: int = 0


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
                label_order=int(seg_kwargs.get("order", 0)),
            ),
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
