from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None

try:
    from scipy.ndimage import gaussian_filter
except ModuleNotFoundError:
    gaussian_filter = None


def _fail_validation(message: str) -> None:
    raise ValueError(message)


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        _fail_validation(f"{name} must be in [0, 1], got {value}")
    return value


def _validate_range(name: str, value: Sequence[float], *, lower: Optional[float] = None) -> Tuple[float, float]:
    if len(value) != 2:
        _fail_validation(f"{name} must contain exactly two values, got {value}")
    result = (float(value[0]), float(value[1]))
    if not all(np.isfinite(i) for i in result) or result[0] > result[1]:
        _fail_validation(f"{name} must be finite and ordered, got {value}")
    if lower is not None and result[0] < lower:
        _fail_validation(f"{name} must be >= {lower}, got {value}")
    return result


def _validate_interpolation(name: str, value: str) -> str:
    value = str(value).lower()
    if value == "bilinear":
        value = "linear"
    if value not in {"nearest", "linear"}:
        _fail_validation(f"{name} must be either 'nearest' or 'linear', got '{value}'")
    return value


def _require_runtime() -> None:
    if torch is None or F is None:
        _fail_validation("NNUNetV2Augmentation requires PyTorch")
    if gaussian_filter is None:
        _fail_validation("NNUNetV2Augmentation requires scipy")


def _is_mask_key(key: str) -> bool:
    return key in {"mask", "mask_a", "mask_b"}


def _is_segmentation_target(key: str, sample: Dict) -> bool:
    if _is_mask_key(key):
        return True
    if key == "target" and (
        sample.get("task_mode") == "segmentation"
        or sample.get("reference_type") == "segmentation"
    ):
        return True
    if key == "evaluation_reference" and (
        sample.get("task_mode") == "segmentation"
        or sample.get("reference_type") == "segmentation"
    ):
        return True
    if key in {"target", "evaluation_reference"}:
        tensor = sample.get(key)
        if torch is not None and isinstance(tensor, torch.Tensor) and not torch.is_floating_point(tensor):
            return True
    return False


def _field_spatial_shape(tensor: "torch.Tensor", key: str) -> Tuple[int, ...]:
    if not isinstance(tensor, torch.Tensor):
        _fail_validation(f"Augmentation field '{key}' must be a torch.Tensor")
    minimum_ndim = 2 if _is_mask_key(key) else 3
    if tensor.ndim < minimum_ndim:
        _fail_validation(f"Augmentation field '{key}' has invalid shape {tuple(tensor.shape)}")
    return tuple(int(i) for i in (tensor.shape if _is_mask_key(key) else tensor.shape[1:]))


def _to_channel_first(tensor: "torch.Tensor", key: str):
    if _is_mask_key(key):
        return tensor.unsqueeze(0), False
    return tensor, True


def _restore_field(tensor: "torch.Tensor", key: str, has_channel: bool) -> "torch.Tensor":
    if not has_channel:
        tensor = tensor[0]
    return tensor


class NNUNetV2Augmentation:
    """Dataset-time augmentation compatible with the nnU-Net v2 defaults.

    This is intentionally implemented without batchgeneratorsv2 so it remains
    usable in this package's Python 3.7 environment. Spatial transforms are
    shared by all fields in a sample. By default, segmentation and binary masks
    use one-hot linear interpolation with ID restoration and a 0.5 threshold;
    nearest interpolation remains available for labels that require it. When a
    dataset supplies a final patch size, ``get_initial_patch_size`` reserves a
    larger temporary patch for spatial augmentation before the final crop.
    """

    def __init__(
        self,
        *,
        p_rotation: float = 0.2,
        p_scaling: float = 0.2,
        scaling: Sequence[float] = (0.7, 1.4),
        p_gaussian_noise: float = 0.1,
        noise_variance: Sequence[float] = (0.0, 0.1),
        p_gaussian_blur: float = 0.2,
        blur_sigma: Sequence[float] = (0.5, 1.0),
        p_brightness: float = 0.15,
        brightness_multiplier: Sequence[float] = (0.75, 1.25),
        p_contrast: float = 0.15,
        contrast_range: Sequence[float] = (0.75, 1.25),
        p_low_resolution: float = 0.25,
        low_resolution_scale: Sequence[float] = (0.5, 1.0),
        p_gamma_invert: float = 0.1,
        p_gamma: float = 0.3,
        gamma_range: Sequence[float] = (0.7, 1.5),
        initial_scale_range: Sequence[float] = (0.85, 1.25),
        label_interpolation: str = "linear",
        mask_interpolation: str = "linear",
        mirror_axes: Optional[Sequence[int]] = None,
        dummy_2d: bool = True,
        anisotropy_threshold: float = 3.0,
        paired_intensity: str = "synchronized",
    ):
        self.p_rotation = _validate_probability("p_rotation", p_rotation)
        self.p_scaling = _validate_probability("p_scaling", p_scaling)
        self.scaling = _validate_range("scaling", scaling, lower=0.0)
        self.p_gaussian_noise = _validate_probability("p_gaussian_noise", p_gaussian_noise)
        self.noise_variance = _validate_range("noise_variance", noise_variance, lower=0.0)
        self.p_gaussian_blur = _validate_probability("p_gaussian_blur", p_gaussian_blur)
        self.blur_sigma = _validate_range("blur_sigma", blur_sigma, lower=0.0)
        self.p_brightness = _validate_probability("p_brightness", p_brightness)
        self.brightness_multiplier = _validate_range("brightness_multiplier", brightness_multiplier, lower=0.0)
        self.p_contrast = _validate_probability("p_contrast", p_contrast)
        self.contrast_range = _validate_range("contrast_range", contrast_range, lower=0.0)
        self.p_low_resolution = _validate_probability("p_low_resolution", p_low_resolution)
        self.low_resolution_scale = _validate_range("low_resolution_scale", low_resolution_scale, lower=0.0)
        self.p_gamma_invert = _validate_probability("p_gamma_invert", p_gamma_invert)
        self.p_gamma = _validate_probability("p_gamma", p_gamma)
        self.gamma_range = _validate_range("gamma_range", gamma_range, lower=0.0)
        self.initial_scale_range = _validate_range("initial_scale_range", initial_scale_range, lower=0.0)
        if self.initial_scale_range[0] <= 0:
            _fail_validation("initial_scale_range must have a positive lower bound")
        self.label_interpolation = _validate_interpolation("label_interpolation", label_interpolation)
        self.mask_interpolation = _validate_interpolation("mask_interpolation", mask_interpolation)
        self.mirror_axes = None if mirror_axes is None else tuple(int(i) for i in mirror_axes)
        self.dummy_2d = bool(dummy_2d)
        self.anisotropy_threshold = float(anisotropy_threshold)
        if not np.isfinite(self.anisotropy_threshold) or self.anisotropy_threshold <= 0:
            _fail_validation("anisotropy_threshold must be finite and positive")
        self.paired_intensity = str(paired_intensity).lower()
        if self.paired_intensity not in {"none", "synchronized"}:
            _fail_validation("paired_intensity must be either 'none' or 'synchronized'")

    def __call__(self, sample: Dict) -> Dict:
        return self.apply(sample)

    def get_initial_patch_size(self, final_patch_size: Sequence[int]) -> Tuple[int, ...]:
        """Return the temporary patch size needed before spatial augmentation."""
        final_patch_size = tuple(int(i) for i in final_patch_size)
        if len(final_patch_size) not in (2, 3) or any(i <= 0 for i in final_patch_size):
            _fail_validation(
                f"final_patch_size must contain positive 2D or 3D values, got {final_patch_size}"
            )
        if self.p_rotation <= 0 and self.p_scaling <= 0:
            return final_patch_size

        if self.p_rotation <= 0:
            max_angle = 0.0
        elif len(final_patch_size) == 2:
            max_angle = 15.0 if max(final_patch_size) / float(min(final_patch_size)) > 1.5 else 180.0
            max_angle = math.radians(max_angle)
        else:
            dummy_2d = (
                self.dummy_2d
                and max(final_patch_size) / float(final_patch_size[0]) > self.anisotropy_threshold
            )
            max_angle = math.radians(180.0 if dummy_2d else 30.0)
        max_angle = min(max_angle, math.radians(90.0))

        vector = np.asarray(final_patch_size, dtype=np.float32)
        required = vector.copy()
        if len(final_patch_size) == 2:
            c = math.cos(max_angle)
            s = math.sin(max_angle)
            rotation = np.asarray(((c, -s), (s, c)), dtype=np.float32)
            required = np.maximum(required, np.abs(vector.dot(rotation)))
        else:
            sx, cx = math.sin(max_angle), math.cos(max_angle)
            sy, cy = math.sin(max_angle), math.cos(max_angle)
            sz, cz = math.sin(max_angle), math.cos(max_angle)
            rotations = (
                np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float32),
                np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float32),
                np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float32),
            )
            for rotation in rotations:
                required = np.maximum(required, np.abs(vector.dot(rotation)))

        scale_min = min(float(self.initial_scale_range[0]), 1.0)
        if self.p_scaling <= 0:
            scale_min = 1.0
        initial = (required / scale_min).astype(int)
        if len(final_patch_size) == 3 and max(final_patch_size) / float(final_patch_size[0]) > self.anisotropy_threshold:
            initial[0] = final_patch_size[0]
        return tuple(max(int(i), int(j)) for i, j in zip(initial, final_patch_size))

    def apply(
        self,
        sample: Dict,
        *,
        rng: Optional[np.random.RandomState] = None,
        crop_size: Optional[Union[Sequence[int], Dict[str, Sequence[int]]]] = None,
    ) -> Dict:
        _require_runtime()
        if not isinstance(sample, dict):
            _fail_validation(f"Augmentation expects a sample dict, got {type(sample).__name__}")
        rng = np.random.RandomState() if rng is None else rng
        if not isinstance(rng, np.random.RandomState):
            _fail_validation("rng must be a numpy RandomState")

        result = dict(sample)
        for fields, is_paired in self._field_groups(result):
            spatial_shape = self._validate_group(result, fields)
            group_crop_size = crop_size
            if isinstance(crop_size, dict):
                if fields and fields[0] not in crop_size:
                    _fail_validation(
                        f"crop_size is missing a value for augmentation group '{fields[0]}'"
                    )
                group_crop_size = crop_size.get(fields[0])
            reference_shape = (
                tuple(int(i) for i in group_crop_size)
                if group_crop_size is not None
                else spatial_shape
            )
            # The planner decides dummy-2D from the requested final patch,
            # not from the enlarged temporary patch.
            dummy_2d = self._use_dummy_2d(reference_shape)
            self._apply_spatial(result, fields, spatial_shape, dummy_2d, rng)
            if group_crop_size is not None:
                self._center_crop_fields(result, fields, group_crop_size)
            transformed_spatial_shape = (
                tuple(int(i) for i in group_crop_size)
                if group_crop_size is not None
                else spatial_shape
            )

            intensity_fields = [
                key for key in fields
                if key in {"image", "image_a", "image_b"}
                or (is_paired and key in {"target", "evaluation_reference"})
            ]
            if is_paired and self.paired_intensity == "none":
                intensity_fields = [key for key in intensity_fields if key in {"image_a", "image_b"}]
            self._apply_intensity(result, intensity_fields, transformed_spatial_shape, dummy_2d, rng)
            self._apply_mirror(result, fields, transformed_spatial_shape, rng)
        return result

    @staticmethod
    def _field_groups(sample: Dict):
        groups = []
        if "image" in sample:
            keys = [
                key for key in (
                    "image", "target", "evaluation_reference", "mask", "conflict_map", "artifact_pred"
                ) if key in sample and sample[key] is not None
            ]
            groups.append((keys, sample.get("reference_type") == "image"))
        if "image_a" in sample:
            keys_a = [key for key in ("image_a", "mask_a") if key in sample and sample[key] is not None]
            groups.append((keys_a, False))
        if "image_b" in sample:
            keys_b = [key for key in ("image_b", "mask_b") if key in sample and sample[key] is not None]
            groups.append((keys_b, False))
        return groups

    @staticmethod
    def _validate_group(sample: Dict, fields: Sequence[str]) -> Tuple[int, ...]:
        if not fields:
            return tuple()
        shapes = [_field_spatial_shape(sample[key], key) for key in fields]
        if any(shape != shapes[0] for shape in shapes[1:]):
            _fail_validation(f"Augmentation fields must share a spatial shape, got {dict(zip(fields, shapes))}")
        if len(shapes[0]) not in (2, 3):
            _fail_validation(f"Only 2D and 3D augmentation is supported, got shape {shapes[0]}")
        return shapes[0]

    def _use_dummy_2d(self, spatial_shape: Sequence[int]) -> bool:
        return (
            self.dummy_2d
            and len(spatial_shape) == 3
            and max(spatial_shape) / float(spatial_shape[0]) > self.anisotropy_threshold
        )

    def _sample_theta(
        self,
        spatial_shape: Sequence[int],
        dummy_2d: bool,
        rng: np.random.RandomState,
    ) -> Optional[np.ndarray]:
        do_rotation = rng.rand() < self.p_rotation
        do_scaling = rng.rand() < self.p_scaling
        if not do_rotation and not do_scaling:
            return None

        scale = float(rng.uniform(*self.scaling)) if do_scaling else 1.0
        if dummy_2d or len(spatial_shape) == 2:
            if len(spatial_shape) == 2 and max(spatial_shape) / float(min(spatial_shape)) > 1.5:
                angle_limit = 15.0
            else:
                angle_limit = 180.0 if dummy_2d else 30.0
            angle = math.radians(float(rng.uniform(-angle_limit, angle_limit))) if do_rotation else 0.0
            c = math.cos(angle)
            s = math.sin(angle)
            rotation = np.array(((c, -s), (s, c)), dtype=np.float32)
            matrix = rotation * scale
            theta = np.zeros((2, 3), dtype=np.float32)
            theta[:, :2] = matrix
            return theta

        angles = [
            math.radians(float(rng.uniform(-30.0, 30.0))) if do_rotation else 0.0
            for _ in range(3)
        ]
        sx, cx = math.sin(angles[0]), math.cos(angles[0])
        sy, cy = math.sin(angles[1]), math.cos(angles[1])
        sz, cz = math.sin(angles[2]), math.cos(angles[2])
        rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float32)
        ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float32)
        rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float32)
        matrix = (rz @ ry @ rx) * scale
        theta = np.zeros((3, 4), dtype=np.float32)
        theta[:, :3] = matrix
        return theta

    def _apply_spatial(
        self,
        sample: Dict,
        fields: Sequence[str],
        spatial_shape: Sequence[int],
        dummy_2d: bool,
        rng: np.random.RandomState,
    ) -> None:
        theta = self._sample_theta(spatial_shape, dummy_2d, rng)
        if theta is not None:
            for key in fields:
                if _is_mask_key(key):
                    interpolation = self.mask_interpolation
                elif _is_segmentation_target(key, sample):
                    interpolation = self.label_interpolation
                else:
                    interpolation = None
                sample[key] = self._affine_field(
                    sample[key],
                    key,
                    interpolation,
                    theta,
                    spatial_shape,
                    dummy_2d,
                )

    def _apply_mirror(
        self,
        sample: Dict,
        fields: Sequence[str],
        spatial_shape: Sequence[int],
        rng: np.random.RandomState,
    ) -> None:
        allowed_axes = self.mirror_axes
        if allowed_axes is None:
            allowed_axes = tuple(range(len(spatial_shape)))
        for axis in allowed_axes:
            if axis < 0 or axis >= len(spatial_shape):
                _fail_validation(f"mirror_axes contains invalid axis {axis} for {len(spatial_shape)}D data")
            if rng.rand() >= 0.5:
                continue
            for key in fields:
                dimension = axis if _is_mask_key(key) else axis + 1
                sample[key] = torch.flip(sample[key], dims=(dimension,))

    @staticmethod
    def _center_crop_fields(sample: Dict, fields: Sequence[str], crop_size: Sequence[int]) -> None:
        crop_size = tuple(int(i) for i in crop_size)
        for key in fields:
            tensor = sample[key]
            spatial_shape = tuple(int(i) for i in (tensor.shape if _is_mask_key(key) else tensor.shape[1:]))
            if len(spatial_shape) != len(crop_size):
                _fail_validation(
                    f"crop_size for '{key}' has {len(crop_size)} dims, expected {len(spatial_shape)}"
                )
            if any(target <= 0 or target > current for target, current in zip(crop_size, spatial_shape)):
                _fail_validation(
                    f"crop_size {crop_size} must fit inside '{key}' spatial shape {spatial_shape}"
                )
            starts = tuple((current - target) // 2 for current, target in zip(spatial_shape, crop_size))
            if _is_mask_key(key):
                slicer = tuple(slice(start, start + target) for start, target in zip(starts, crop_size))
            else:
                slicer = (slice(None),) + tuple(
                    slice(start, start + target) for start, target in zip(starts, crop_size)
                )
            sample[key] = tensor[slicer].contiguous()

    def _affine_field(
        self,
        tensor: "torch.Tensor",
        key: str,
        interpolation: Optional[str],
        theta_array: np.ndarray,
        spatial_shape: Sequence[int],
        dummy_2d: bool,
    ) -> "torch.Tensor":
        channel_first, has_channel = _to_channel_first(tensor, key)
        input_dtype = channel_first.dtype
        input_device = channel_first.device
        data = channel_first.float().unsqueeze(0)
        grid = self._build_grid(
            channels=int(channel_first.shape[0]),
            spatial_shape=spatial_shape,
            dummy_2d=dummy_2d,
            theta_array=theta_array,
            device=input_device,
        )

        if interpolation == "linear":
            transformed = self._affine_label_field(
                channel_first,
                grid,
                spatial_shape,
                dummy_2d,
                preserve_partial=not _is_mask_key(key),
            )
        elif interpolation == "nearest":
            transformed = self._grid_sample(
                data,
                grid,
                spatial_shape,
                dummy_2d,
                mode="nearest",
            )[0]
        else:
            transformed = self._grid_sample(
                data,
                grid,
                spatial_shape,
                dummy_2d,
                mode="linear",
            )[0]

        if input_dtype == torch.bool:
            transformed = transformed > 0.5
        elif not torch.is_floating_point(torch.empty((), dtype=input_dtype)):
            transformed = torch.round(transformed).to(input_dtype)
        else:
            transformed = transformed.to(input_dtype)
        return _restore_field(transformed, key, has_channel)

    @staticmethod
    def _build_grid(
        *,
        channels: int,
        spatial_shape: Sequence[int],
        dummy_2d: bool,
        theta_array: np.ndarray,
        device,
    ) -> "torch.Tensor":
        if dummy_2d:
            output_shape = (1, channels * int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2]))
            axis_sizes = np.asarray(spatial_shape[1:], dtype=np.float32)
        else:
            output_shape = (1, channels, *tuple(int(i) for i in spatial_shape))
            axis_sizes = np.asarray(spatial_shape, dtype=np.float32)
        affine = np.asarray(theta_array, dtype=np.float32)
        linear = affine[:, :-1]
        normalized = np.empty_like(affine)
        normalized[:, :-1] = np.diag(1.0 / axis_sizes).dot(linear).dot(np.diag(axis_sizes))
        # affine_grid requires (spatial_dims, spatial_dims + 1). Keep the
        # translation column rather than reducing theta to a square matrix.
        normalized[:, -1] = affine[:, -1]
        theta = torch.as_tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
        return F.affine_grid(theta, output_shape, align_corners=False)

    @staticmethod
    def _grid_sample(
        data: "torch.Tensor",
        grid: "torch.Tensor",
        spatial_shape: Sequence[int],
        dummy_2d: bool,
        *,
        mode: str,
    ) -> "torch.Tensor":
        if mode == "linear":
            # grid_sample calls 5D trilinear interpolation "bilinear" too.
            # "trilinear" is only a valid mode name for interpolate.
            mode = "bilinear"
        if dummy_2d:
            channels = int(data.shape[1])
            depth = int(spatial_shape[0])
            data_2d = data.reshape(1, channels * depth, spatial_shape[1], spatial_shape[2])
            transformed = F.grid_sample(
                data_2d,
                grid,
                mode=mode,
                padding_mode="zeros",
                align_corners=False,
            )
            return transformed.reshape(1, channels, depth, spatial_shape[1], spatial_shape[2])
        return F.grid_sample(
            data,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        )

    def _affine_label_field(
        self,
        channel_first: "torch.Tensor",
        grid: "torch.Tensor",
        spatial_shape: Sequence[int],
        dummy_2d: bool,
        preserve_partial: bool,
    ) -> "torch.Tensor":
        """Interpolate each label independently and resolve overlaps without losing IDs."""
        result = torch.zeros_like(channel_first)
        for channel in range(int(channel_first.shape[0])):
            values = channel_first[channel]
            labels = torch.sort(torch.unique(values)).values
            best_score = torch.zeros(spatial_shape, dtype=torch.float32, device=values.device)
            best_any_score = torch.zeros_like(best_score)
            best_any_label = torch.zeros_like(values)
            restored = torch.zeros_like(values)
            for label in labels:
                if label.item() == 0:
                    continue
                one_hot = (values == label).float().unsqueeze(0).unsqueeze(0)
                score = self._grid_sample(
                    one_hot,
                    grid,
                    spatial_shape,
                    dummy_2d,
                    mode="linear",
                )[0, 0]
                better_any = score > best_any_score
                best_any_score = torch.where(better_any, score, best_any_score)
                best_any_label = torch.where(better_any, label.to(dtype=best_any_label.dtype), best_any_label)
                candidate = score >= 0.5
                better = candidate & (score > best_score)
                restored = torch.where(better, label.to(dtype=restored.dtype), restored)
                best_score = torch.where(better, score, best_score)
            if preserve_partial:
                fallback = (restored == 0) & (best_any_score > 0)
                restored = torch.where(fallback, best_any_label, restored)
            result[channel] = restored
        return result

    def _apply_intensity(
        self,
        sample: Dict,
        fields: Sequence[str],
        spatial_shape: Sequence[int],
        dummy_2d: bool,
        rng: np.random.RandomState,
    ) -> None:
        fields = [key for key in fields if torch.is_floating_point(sample[key])]
        if not fields:
            return

        if rng.rand() < self.p_gaussian_noise:
            variance = float(rng.uniform(*self.noise_variance))
            shared_noise = rng.normal(0.0, math.sqrt(variance), size=tuple(spatial_shape)).astype(np.float32)
            for key in fields:
                tensor = sample[key]
                noise = torch.as_tensor(shared_noise, dtype=tensor.dtype, device=tensor.device)
                sample[key] = tensor + noise.unsqueeze(0)

        if rng.rand() < self.p_gaussian_blur:
            sigma = tuple(float(rng.uniform(*self.blur_sigma)) for _ in spatial_shape)
            for key in fields:
                sample[key] = self._blur(sample[key], sigma)

        if rng.rand() < self.p_brightness:
            multiplier = float(rng.uniform(*self.brightness_multiplier))
            for key in fields:
                sample[key] = sample[key] * multiplier

        if rng.rand() < self.p_contrast:
            factor = float(rng.uniform(*self.contrast_range))
            for key in fields:
                tensor = sample[key]
                mean = tensor.mean(dim=tuple(range(1, tensor.ndim)), keepdim=True)
                sample[key] = mean + (tensor - mean) * factor

        if rng.rand() < self.p_low_resolution:
            scale = float(rng.uniform(*self.low_resolution_scale))
            if scale < 1.0:
                for key in fields:
                    sample[key] = self._simulate_low_resolution(sample[key], scale, dummy_2d)

        if rng.rand() < self.p_gamma_invert:
            gamma = float(rng.uniform(*self.gamma_range))
            for key in fields:
                sample[key] = self._gamma(sample[key], gamma, invert=True)

        if rng.rand() < self.p_gamma:
            gamma = float(rng.uniform(*self.gamma_range))
            for key in fields:
                sample[key] = self._gamma(sample[key], gamma, invert=False)

    @staticmethod
    def _blur(tensor: "torch.Tensor", sigma: Sequence[float]) -> "torch.Tensor":
        array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        result = np.empty_like(array)
        for channel in range(array.shape[0]):
            result[channel] = gaussian_filter(array[channel], sigma=sigma, mode="nearest")
        return torch.from_numpy(np.ascontiguousarray(result)).to(device=tensor.device, dtype=tensor.dtype)

    @staticmethod
    def _simulate_low_resolution(tensor: "torch.Tensor", scale: float, dummy_2d: bool) -> "torch.Tensor":
        spatial_shape = tuple(int(i) for i in tensor.shape[1:])
        if dummy_2d and len(spatial_shape) == 3:
            data = tensor.reshape(tensor.shape[0] * spatial_shape[0], 1, spatial_shape[1], spatial_shape[2])
            size = (max(1, int(round(spatial_shape[1] * scale))), max(1, int(round(spatial_shape[2] * scale))))
            small = F.interpolate(data, size=size, mode="bilinear", align_corners=False)
            restored = F.interpolate(small, size=(spatial_shape[1], spatial_shape[2]), mode="bilinear", align_corners=False)
            return restored.reshape(tensor.shape[0], *spatial_shape)
        data = tensor.unsqueeze(0)
        size = tuple(max(1, int(round(i * scale))) for i in spatial_shape)
        mode = "bilinear" if len(spatial_shape) == 2 else "trilinear"
        small = F.interpolate(data, size=size, mode=mode, align_corners=False)
        restored = F.interpolate(small, size=spatial_shape, mode=mode, align_corners=False)
        return restored[0]

    @staticmethod
    def _gamma(tensor: "torch.Tensor", gamma: float, *, invert: bool) -> "torch.Tensor":
        result = tensor.clone()
        for channel in range(tensor.shape[0]):
            values = tensor[channel]
            minimum = values.min()
            maximum = values.max()
            if (
                not bool(torch.isfinite(minimum).item())
                or not bool(torch.isfinite(maximum).item())
                or not bool((maximum > minimum).item())
            ):
                continue
            mean = values.mean()
            std = values.std(unbiased=False)
            normalized = (values - minimum) / (maximum - minimum)
            if invert:
                normalized = 1.0 - normalized
            transformed = normalized.clamp(0.0, 1.0).pow(gamma)
            transformed = transformed * (maximum - minimum) + minimum
            new_mean = transformed.mean()
            new_std = transformed.std(unbiased=False)
            if bool((std > 0).item()) and bool((new_std > 0).item()):
                transformed = (transformed - new_mean) / new_std * std + mean
            result[channel] = transformed
        return result
