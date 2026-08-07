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
from skimage.transform import resize


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


def _resize_segmentation(segmentation: np.ndarray, new_shape: Sequence[int], order: int = 1) -> np.ndarray:
    if order == 0:
        return resize(
            segmentation,
            new_shape,
            order=0,
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        ).astype(segmentation.dtype, copy=False)

    result = np.zeros(new_shape, dtype=segmentation.dtype)
    for label in np.unique(segmentation):
        resized_mask = resize(
            (segmentation == label).astype(float),
            new_shape,
            order=order,
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        )
        result[resized_mask >= 0.5] = label
    return result


def _resize_data_or_seg(
    data: np.ndarray,
    new_shape: Sequence[int],
    *,
    is_seg: bool,
    order: int,
) -> np.ndarray:
    if is_seg:
        return _resize_segmentation(data, new_shape, order=order)
    return resize(
        data,
        new_shape,
        order=order,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )


def _resample_data_or_seg(
    data: np.ndarray,
    new_shape: Sequence[int],
    *,
    is_seg: bool,
    orders: Sequence[int],
    dtype_out=None,
) -> np.ndarray:
    if data.ndim not in (3, 4):
        _fail_validation(f"data must be (C, X, Y) or (C, X, Y, Z), got {data.shape}")
    if len(new_shape) != data.ndim - 1:
        _fail_validation(f"new_shape must match spatial dims, got {new_shape} for data shape {data.shape}")

    new_shape = tuple(int(i) for i in new_shape)
    orders = tuple(int(i) for i in orders)
    if len(orders) != len(new_shape):
        _fail_validation(
            f"orders must contain {len(new_shape)} values for the spatial dimensions, got {len(orders)}"
        )
    if any(order < 0 or order > MAX_INTERPOLATION_ORDER for order in orders):
        _fail_validation(
            f"orders must be between 0 and {MAX_INTERPOLATION_ORDER}, got {orders}"
        )

    shape = tuple(int(i) for i in data.shape[1:])
    if dtype_out is None:
        dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    if shape == new_shape:
        return data

    data = data.astype(float, copy=False)
    if len(set(orders)) == 1:
        for c in range(data.shape[0]):
            reshaped_final[c] = _resize_data_or_seg(
                data[c],
                new_shape,
                is_seg=is_seg,
                order=orders[0],
            )
        return reshaped_final

    # Different orders are applied one spatial axis at a time. This keeps the
    # interpolation policy explicit for every axis instead of treating one
    # axis as a special "z" dimension.
    for c in range(data.shape[0]):
        resampled = data[c]
        for axis, order in enumerate(orders):
            if resampled.shape[axis] == new_shape[axis]:
                continue
            axis_shape = list(resampled.shape)
            axis_shape[axis] = new_shape[axis]
            resampled = _resize_data_or_seg(
                resampled,
                tuple(axis_shape),
                is_seg=is_seg,
                order=order,
            )
        reshaped_final[c] = resampled
    return reshaped_final


def resample_array(
    data: np.ndarray,
    new_shape: Sequence[int],
    *,
    is_seg: bool,
    orders: Optional[Sequence[int]] = None,
    order: Optional[int] = None,
) -> np.ndarray:
    if len(new_shape) != data.ndim - 1:
        _fail_validation(
            f"new_shape must match data spatial dims, got {len(new_shape)} and data shape {data.shape}"
        )
    if any(i <= 0 for i in new_shape):
        _fail_validation(f"new_shape must contain only positive values, got {tuple(new_shape)}")
    if orders is not None and order is not None:
        _fail_validation("Provide either orders or order, not both")
    if orders is None:
        if order is None:
            _fail_validation("orders must be provided")
        orders = tuple(int(order) for _ in range(len(new_shape)))
    return _resample_data_or_seg(
        data,
        new_shape,
        is_seg=is_seg,
        orders=orders,
    )
