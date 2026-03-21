from __future__ import annotations

from typing import Dict, Type
import warnings

import numpy as np


class ImageNormalization:
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False
    required_intensity_properties: tuple[str, ...] = ()

    def __init__(self, use_mask_for_norm: bool = False, intensity_properties: dict | None = None):
        self.use_mask_for_norm = bool(use_mask_for_norm)
        self.intensity_properties = intensity_properties or {}

    def run(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        eps = 1e-8
        if self.use_mask_for_norm and mask is not None and np.any(mask):
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / max(std, eps)
            return image
        mean = image.mean()
        std = image.std()
        image -= mean
        image /= max(std, eps)
        return image


class CTNormalization(ImageNormalization):
    required_intensity_properties = ("percentile_00_5", "percentile_99_5", "mean", "std")

    def run(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        eps = 1e-8
        lower = self.intensity_properties.get("clip_min", self.intensity_properties["percentile_00_5"])
        upper = self.intensity_properties.get("clip_max", self.intensity_properties["percentile_99_5"])
        mean = self.intensity_properties["mean"]
        std = self.intensity_properties["std"]
        np.clip(image, lower, upper, out=image)
        image -= mean
        image /= max(std, eps)
        return image


class MinMaxClipNormalization(ImageNormalization):
    required_intensity_properties = ("clip_min", "clip_max")

    def run(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        eps = 1e-8
        lower = self.intensity_properties["clip_min"]
        upper = self.intensity_properties["clip_max"]
        if lower is None or upper is None:
            _fail_validation("MinMaxClipNormalization requires non-null clip_min and clip_max")
        lower = float(lower)
        upper = float(upper)
        if lower >= upper:
            _fail_validation(
                f"MinMaxClipNormalization requires clip_min < clip_max, got {lower} and {upper}"
            )
        np.clip(image, lower, upper, out=image)
        image -= lower
        image /= max(upper - lower, eps)
        return image


class NoNormalization(ImageNormalization):
    def run(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return image.astype(np.float32, copy=False)


class RescaleTo01Normalization(ImageNormalization):
    def run(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        eps = 1e-8
        image -= image.min()
        image /= max(float(image.max()), eps)
        return image


class RGBTo01Normalization(ImageNormalization):
    def run(self, image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        image = image.astype(np.float32, copy=False)
        image /= 255.0
        return image


NORMALIZATION_REGISTRY: Dict[str, Type[ImageNormalization]] = {
    "zscore": ZScoreNormalization,
    "zscorenormalization": ZScoreNormalization,
    "ctnormalization": CTNormalization,
    "ct": CTNormalization,
    "minmaxclipnormalization": MinMaxClipNormalization,
    "minmax_clip": MinMaxClipNormalization,
    "minmaxclip": MinMaxClipNormalization,
    "nonormalization": NoNormalization,
    "nonorm": NoNormalization,
    "rescaleto01normalization": RescaleTo01Normalization,
    "rescale_to_0_1": RescaleTo01Normalization,
    "rgbto01normalization": RGBTo01Normalization,
    "rgb_to_0_1": RGBTo01Normalization,
}


def _fail_validation(message: str) -> None:
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)


def get_normalizer(name: str) -> Type[ImageNormalization]:
    key = name.casefold()
    if key not in NORMALIZATION_REGISTRY:
        _fail_validation(f"Unsupported normalization scheme '{name}'")
    return NORMALIZATION_REGISTRY[key]
