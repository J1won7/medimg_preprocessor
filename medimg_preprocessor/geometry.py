from __future__ import annotations

from typing import Optional, Sequence
import warnings

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

try:
    import SimpleITK as sitk
except (ImportError, OSError):  # pragma: no cover - exercised by installation failures
    sitk = None


MAX_INTERPOLATION_ORDER = 5
POSTPROCESS_CHOICES = (
    "none",
    "fill_holes",
    "closing",
    "fill_holes_closing",
)


def _fail_validation(message: str) -> None:
    warnings.warn(message, stacklevel=2)
    raise ValueError(message)


def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    if data.ndim not in (3, 4):
        _fail_validation(f"Expected data with shape (C, X, Y) or (C, X, Y, Z), got {data.shape}")
    mask = data[0] != 0
    for c in range(1, data.shape[0]):
        mask |= data[c] != 0
    return binary_fill_holes(mask)


def create_threshold_mask(data: np.ndarray, threshold: float) -> np.ndarray:
    if data.ndim not in (3, 4):
        _fail_validation(f"Expected data with shape (C, X, Y) or (C, X, Y, Z), got {data.shape}")
    mask = data[0] > float(threshold)
    for c in range(1, data.shape[0]):
        mask |= data[c] > float(threshold)
    return mask


def ensure_binary_mask(mask: np.ndarray, *, spatial_shape: Sequence[int], name: str = "mask") -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        _fail_validation(f"{name} must be a numpy.ndarray, got {type(mask).__name__}")
    if mask.ndim == len(spatial_shape):
        reduced = mask
    elif mask.ndim == len(spatial_shape) + 1:
        if tuple(mask.shape[1:]) != tuple(spatial_shape):
            _fail_validation(
                f"{name} spatial shape must match image, got {mask.shape[1:]} and {tuple(spatial_shape)}"
            )
        reduced = np.any(mask != 0, axis=0)
    else:
        _fail_validation(
            f"{name} must have shape {tuple(spatial_shape)} or (C, {', '.join(str(i) for i in spatial_shape)}), got {mask.shape}"
        )
    if tuple(reduced.shape) != tuple(spatial_shape):
        _fail_validation(f"{name} spatial shape must match image, got {reduced.shape} and {tuple(spatial_shape)}")
    return np.asarray(reduced != 0, dtype=bool)


def _apply_morphology(
    mask: np.ndarray,
    *,
    operation: str,
    closing_iters: int,
) -> np.ndarray:
    if operation not in POSTPROCESS_CHOICES:
        _fail_validation(
            f"Unknown postprocess operation '{operation}'. "
            f"Choose one of {POSTPROCESS_CHOICES}"
        )
    iterations = int(closing_iters)
    if iterations < 0:
        _fail_validation(f"closing_iters must be non-negative, got {closing_iters}")

    result = np.asarray(mask, dtype=bool)
    structure = generate_binary_structure(result.ndim, 1)
    use_fill_holes = operation in {"fill_holes", "fill_holes_closing"}
    use_closing = operation in {"closing", "fill_holes_closing"}
    if use_fill_holes:
        result = binary_fill_holes(result, structure=structure)
    if use_closing and iterations > 0:
        padding = [(iterations, iterations)] * result.ndim
        padded = np.pad(result, padding, mode="constant", constant_values=False)
        closed = binary_closing(padded, structure=structure, iterations=iterations)
        crop_slices = tuple(slice(iterations, -iterations) for _ in range(result.ndim))
        result = closed[crop_slices]
    if use_fill_holes:
        result = binary_fill_holes(result, structure=structure)
    return np.asarray(result, dtype=bool)


def postprocess_binary_mask(
    mask: np.ndarray,
    *,
    operation: str = "none",
    closing_iters: int = 1,
    keep_largest_component: bool = False,
) -> np.ndarray:
    """Apply the single final-stage morphology pass to a binary mask."""
    mask = np.asarray(mask)
    if mask.ndim not in (2, 3):
        _fail_validation(
            f"mask must be a 2D or 3D array, got shape {mask.shape}"
        )
    result = _apply_morphology(
        mask,
        operation=operation,
        closing_iters=closing_iters,
    )
    if keep_largest_component and np.any(result):
        structure = generate_binary_structure(result.ndim, 1)
        labeled, num = label(result, structure=structure)
        if num > 1:
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0
            largest = int(np.argmax(component_sizes))
            result = labeled == largest
    return np.asarray(result, dtype=bool)


def postprocess_label_instances(
    label_map: np.ndarray,
    *,
    operation: str = "none",
    closing_iters: int = 1,
    spacing: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Apply morphology independently to each positive label ID.

    Original non-background voxels are protected. If multiple processed
    instances claim the same newly created voxel, the instance with the
    smallest physical distance to its original mask wins. Ties use the
    smaller label ID for deterministic output.
    """
    if not isinstance(label_map, np.ndarray) or label_map.ndim not in (2, 3):
        _fail_validation(
            f"label_map must be a 2D or 3D numpy array, got {getattr(label_map, 'shape', None)}"
        )
    if not np.issubdtype(label_map.dtype, np.number):
        _fail_validation(f"label_map must contain numeric data, got {label_map.dtype}")
    if operation not in POSTPROCESS_CHOICES:
        _fail_validation(
            f"Unknown postprocess operation '{operation}'. "
            f"Choose one of {POSTPROCESS_CHOICES}"
        )
    if operation == "none":
        return np.asarray(label_map)
    iterations = int(closing_iters)
    if iterations < 0:
        _fail_validation(f"closing_iters must be non-negative, got {closing_iters}")

    sampling = None
    if spacing is not None:
        sampling = tuple(float(value) for value in spacing)
        if len(sampling) != label_map.ndim or any(
            not np.isfinite(value) or value <= 0 for value in sampling
        ):
            _fail_validation(
                f"spacing must contain {label_map.ndim} finite positive values, got {spacing}"
            )

    original = np.asarray(label_map).copy()
    occupied = original != 0
    best_distance = np.full(original.shape, np.inf, dtype=np.float64)
    best_label = np.zeros(original.shape, dtype=original.dtype)
    for label_value in np.unique(original):
        if label_value <= 0:
            continue
        instance = original == label_value
        processed = _apply_morphology(
            instance,
            operation=operation,
            closing_iters=iterations,
        )
        # Never overwrite an existing voxel belonging to another label.
        processed &= (~occupied) | instance
        new_voxels = processed & ~occupied
        if not np.any(new_voxels):
            continue
        distances = distance_transform_edt(~instance, sampling=sampling)
        better = new_voxels & (
            (distances < best_distance)
            | (
                (distances == best_distance)
                & ((best_label == 0) | (label_value < best_label))
            )
        )
        best_distance[better] = distances[better]
        best_label[better] = label_value

    result = original.copy()
    assigned = best_label != 0
    result[assigned] = best_label[assigned]
    return result


def compute_new_shape(
    old_shape: Sequence[int],
    old_spacing: Sequence[float],
    new_spacing: Sequence[float],
) -> np.ndarray:
    if len(old_shape) != len(old_spacing) or len(old_shape) != len(new_spacing):
        _fail_validation(
            "old_shape, old_spacing, and new_spacing must have identical length, "
            f"got {len(old_shape)}, {len(old_spacing)}, {len(new_spacing)}"
        )
    if any(i <= 0 for i in old_shape):
        _fail_validation(f"old_shape must contain only positive values, got {tuple(old_shape)}")
    if any(i <= 0 for i in old_spacing) or any(i <= 0 for i in new_spacing):
        _fail_validation(
            f"old_spacing and new_spacing must contain only positive values, got {tuple(old_spacing)} and {tuple(new_spacing)}"
    )
    return np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])


_MASK_LINEAR_FALLBACK_WARNED = False


def _require_simpleitk() -> None:
    if sitk is None:
        _fail_validation(
            "SimpleITK is required for image and mask resampling; install the package dependencies first"
        )


def _identity_direction(spatial_dims: int) -> tuple:
    return tuple(float(value) for value in np.eye(spatial_dims, dtype=np.float64).ravel())


def _normalize_spacing(
    spacing: Optional[Sequence[float]],
    spatial_dims: int,
    name: str,
) -> tuple:
    if spacing is None:
        return (1.0,) * spatial_dims
    if len(spacing) != spatial_dims:
        _fail_validation(f"{name} must contain {spatial_dims} values")
    values = tuple(float(value) for value in spacing)
    if any(not np.isfinite(value) or value <= 0 for value in values):
        _fail_validation(f"{name} must contain finite positive values")
    return values


def _normalize_orders(
    data: np.ndarray,
    orders: Optional[Sequence[int]],
    order: Optional[int],
) -> tuple:
    spatial_dims = data.ndim - 1
    if orders is not None and order is not None:
        _fail_validation("provide either orders or order, not both")
    if orders is None:
        if order is None:
            _fail_validation("an interpolation order is required")
        orders = (int(order),) * spatial_dims
    if len(orders) != spatial_dims:
        _fail_validation(
            f"orders must contain {spatial_dims} values for data with {spatial_dims} spatial dimensions"
        )
    normalized = tuple(int(value) for value in orders)
    if any(value < 0 or value > MAX_INTERPOLATION_ORDER for value in normalized):
        _fail_validation(
            f"resampling orders must be between 0 and {MAX_INTERPOLATION_ORDER}"
        )
    return normalized


def _validate_resampling_inputs(
    data: np.ndarray,
    new_shape: Sequence[int],
    current_spacing: Optional[Sequence[float]],
    new_spacing: Optional[Sequence[float]],
    orders: Optional[Sequence[int]],
    order: Optional[int],
) -> tuple:
    if data.ndim not in (3, 4):
        _fail_validation(
            f"resampling expects channel-first 2D/3D data (C, X, Y[, Z]), got shape {data.shape}"
        )
    spatial_dims = data.ndim - 1
    target_shape = tuple(int(value) for value in new_shape)
    if len(target_shape) != spatial_dims or any(value <= 0 for value in target_shape):
        _fail_validation(f"new_shape must contain {spatial_dims} positive values")
    if not np.all(np.isfinite(data)):
        _fail_validation("resampling input contains non-finite values")

    source_spacing = _normalize_spacing(current_spacing, spatial_dims, "current_spacing")
    if new_spacing is None:
        target_spacing = tuple(
            source_spacing[axis] * float(data.shape[axis + 1]) / float(target_shape[axis])
            for axis in range(spatial_dims)
        )
    else:
        target_spacing = _normalize_spacing(new_spacing, spatial_dims, "new_spacing")
    normalized_orders = _normalize_orders(data, orders, order)
    return target_shape, source_spacing, target_spacing, normalized_orders


def _simpleitk_interpolator(role: str, order: int):
    _require_simpleitk()
    if role == "mask":
        if order == 0:
            return sitk.sitkNearestNeighbor
        if order != 1:
            _fail_validation("mask interpolation supports only nearest (0) or label-linear (1)")

        label_linear = getattr(sitk, "sitkLabelLinear", None)
        if label_linear is not None:
            return label_linear

        # SimpleITK 2.1.1, required for Python 3.7, predates sitkLabelLinear.
        # LabelGaussian is label-aware and preserves the original label set.
        label_gaussian = getattr(sitk, "sitkLabelGaussian", None)
        if label_gaussian is not None:
            global _MASK_LINEAR_FALLBACK_WARNED
            if not _MASK_LINEAR_FALLBACK_WARNED:
                warnings.warn(
                    "sitkLabelLinear is unavailable; mask order 1 uses "
                    "sitkLabelGaussian for compatibility with this SimpleITK version",
                    RuntimeWarning,
                    stacklevel=3,
                )
                _MASK_LINEAR_FALLBACK_WARNED = True
            return label_gaussian
        _fail_validation(
            "mask order 1 requires sitkLabelLinear or sitkLabelGaussian; use mask order 0 "
            "with this SimpleITK installation"
        )

    if role != "image":
        _fail_validation(f"unsupported resampling role: {role}")
    if order == 0:
        return sitk.sitkNearestNeighbor
    if order == 1:
        return sitk.sitkLinear
    interpolator = getattr(sitk, "sitkBSpline" + str(order), None)
    if interpolator is None:
        _fail_validation(f"SimpleITK does not provide a B-spline order {order} interpolator")
    return interpolator


def _array_to_sitk_image(array: np.ndarray, spacing: Sequence[float]):
    spatial_dims = array.ndim
    image = sitk.GetImageFromArray(
        np.transpose(array, tuple(range(spatial_dims - 1, -1, -1)))
    )
    image.SetSpacing(tuple(float(value) for value in spacing))
    image.SetOrigin((0.0,) * spatial_dims)
    image.SetDirection(_identity_direction(spatial_dims))
    return image


def _array_to_sitk_vector_image(array: np.ndarray, spacing: Sequence[float]):
    spatial_dims = array.ndim - 1
    channel_last = np.moveaxis(array, 0, -1)
    itk_array = np.transpose(
        channel_last,
        tuple(range(spatial_dims - 1, -1, -1)) + (spatial_dims,),
    )
    image = sitk.GetImageFromArray(itk_array, isVector=True)
    image.SetSpacing(tuple(float(value) for value in spacing))
    image.SetOrigin((0.0,) * spatial_dims)
    image.SetDirection(_identity_direction(spatial_dims))
    return image


def _resample_sitk_image(
    image,
    new_shape: Sequence[int],
    new_spacing: Sequence[float],
    interpolator,
    use_nearest_extrapolator: bool,
):
    spatial_dims = len(new_shape)
    resampler = sitk.ResampleImageFilter()
    # The input array is reversed before GetImageFromArray, so ITK's x/y/z
    # image size is already expressed in the package's spatial-axis order.
    resampler.SetSize(tuple(int(value) for value in new_shape))
    resampler.SetOutputSpacing(tuple(float(value) for value in new_spacing))
    resampler.SetOutputOrigin((0.0,) * spatial_dims)
    resampler.SetOutputDirection(_identity_direction(spatial_dims))
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(interpolator)
    if use_nearest_extrapolator and hasattr(resampler, "SetUseNearestNeighborExtrapolator"):
        resampler.SetUseNearestNeighborExtrapolator(True)
    return resampler.Execute(image)


def _sitk_image_to_array(image) -> np.ndarray:
    array = np.asarray(sitk.GetArrayFromImage(image))
    return np.transpose(array, tuple(range(array.ndim - 1, -1, -1)))


def _sitk_vector_image_to_array(image, spatial_dims: int) -> np.ndarray:
    array = np.asarray(sitk.GetArrayFromImage(image))
    array = np.transpose(array, tuple(range(spatial_dims - 1, -1, -1)) + (spatial_dims,))
    return np.moveaxis(array, -1, 0)


def _coerce_label_input(data: np.ndarray) -> np.ndarray:
    rounded = np.rint(data)
    if not np.array_equal(data, rounded):
        _fail_validation("mask/label resampling input must contain integer label values")
    info = np.iinfo(np.int32)
    if np.any(rounded < info.min) or np.any(rounded > info.max):
        _fail_validation("mask/label values must fit in a signed 32-bit integer")
    return rounded.astype(np.int32, copy=False)


def _resample_once(
    data: np.ndarray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    role: str,
    order: int,
) -> np.ndarray:
    interpolator = _simpleitk_interpolator(role, order)
    if role == "image":
        working = data.astype(np.float32, copy=False)
        if working.shape[0] == 1:
            image = _array_to_sitk_image(working[0], current_spacing)
            output = _resample_sitk_image(
                image,
                new_shape,
                new_spacing,
                interpolator,
                use_nearest_extrapolator=True,
            )
            return _sitk_image_to_array(output)[None]
        image = _array_to_sitk_vector_image(working, current_spacing)
        output = _resample_sitk_image(
            image,
            new_shape,
            new_spacing,
            interpolator,
            use_nearest_extrapolator=True,
        )
        return _sitk_vector_image_to_array(output, data.ndim - 1)

    working = _coerce_label_input(data)
    result = np.empty((working.shape[0],) + tuple(new_shape), dtype=np.int32)
    for channel in range(working.shape[0]):
        image = _array_to_sitk_image(working[channel], current_spacing)
        output = _resample_sitk_image(
            image,
            new_shape,
            new_spacing,
            interpolator,
            use_nearest_extrapolator=True,
        )
        result[channel] = _sitk_image_to_array(output)
    return result


def _resample_role(
    data: np.ndarray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    orders: Sequence[int],
    role: str,
) -> np.ndarray:
    if role == "mask" and any(order not in (0, 1) for order in orders):
        _fail_validation("mask interpolation supports only nearest (0) or label-linear (1)")

    target_shape = tuple(int(value) for value in new_shape)
    if tuple(data.shape[1:]) == target_shape and np.allclose(current_spacing, new_spacing):
        return data.copy()

    if all(order == orders[0] for order in orders):
        return _resample_once(
            data,
            target_shape,
            current_spacing,
            new_spacing,
            role,
            orders[0],
        ).astype(data.dtype, copy=False)

    # SimpleITK applies one interpolator per operation. Explicit per-axis
    # policies therefore use sequential physical-space resampling. The common
    # same-order path above remains a single operation.
    result = data
    intermediate_shape = list(data.shape[1:])
    intermediate_spacing = list(current_spacing)
    for axis, axis_order in enumerate(orders):
        if (
            intermediate_shape[axis] == target_shape[axis]
            and np.isclose(intermediate_spacing[axis], new_spacing[axis])
        ):
            continue
        step_source_spacing = tuple(intermediate_spacing)
        intermediate_shape[axis] = target_shape[axis]
        intermediate_spacing[axis] = new_spacing[axis]
        result = _resample_once(
            result,
            tuple(intermediate_shape),
            step_source_spacing,
            tuple(intermediate_spacing),
            role,
            axis_order,
        ).astype(data.dtype, copy=False)
    return result


def _resample_public(
    data: np.ndarray,
    new_shape: Sequence[int],
    role: str,
    current_spacing: Optional[Sequence[float]],
    new_spacing: Optional[Sequence[float]],
    orders: Optional[Sequence[int]],
    order: Optional[int],
) -> np.ndarray:
    data = np.asarray(data)
    target_shape, source_spacing, target_spacing, normalized_orders = _validate_resampling_inputs(
        data,
        new_shape,
        current_spacing,
        new_spacing,
        orders,
        order,
    )
    return _resample_role(
        data,
        target_shape,
        source_spacing,
        target_spacing,
        normalized_orders,
        role,
    )


def resample_image(
    data: np.ndarray,
    new_shape: Sequence[int],
    current_spacing: Optional[Sequence[float]] = None,
    new_spacing: Optional[Sequence[float]] = None,
    orders: Optional[Sequence[int]] = None,
    order: Optional[int] = None,
) -> np.ndarray:
    """Resample channel-first continuous image data with SimpleITK."""

    return _resample_public(
        data,
        new_shape,
        "image",
        current_spacing,
        new_spacing,
        orders,
        order,
    )


def resample_mask(
    data: np.ndarray,
    new_shape: Sequence[int],
    current_spacing: Optional[Sequence[float]] = None,
    new_spacing: Optional[Sequence[float]] = None,
    orders: Optional[Sequence[int]] = None,
    order: Optional[int] = None,
) -> np.ndarray:
    """Resample channel-first masks/labels with SimpleITK label interpolators."""

    return _resample_public(
        data,
        new_shape,
        "mask",
        current_spacing,
        new_spacing,
        orders,
        order,
    )


def resample_array(
    data: np.ndarray,
    new_shape: Sequence[int],
    *,
    is_seg: bool,
    orders: Optional[Sequence[int]] = None,
    order: Optional[int] = None,
    current_spacing: Optional[Sequence[float]] = None,
    new_spacing: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Backward-compatible dispatcher for image and mask resampling."""

    function = resample_mask if is_seg else resample_image
    return function(
        data,
        new_shape,
        current_spacing=current_spacing,
        new_spacing=new_spacing,
        orders=orders,
        order=order,
    )
