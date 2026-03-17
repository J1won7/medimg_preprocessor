from __future__ import annotations

from typing import List, Sequence, Tuple
import warnings

import numpy as np
from scipy.ndimage import binary_fill_holes, zoom


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


def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return [[0, int(i)] for i in mask.shape]
    return [[int(c.min()), int(c.max()) + 1] for c in coords]


def bbox_to_slices(bbox: Sequence[Sequence[int]]) -> Tuple[slice, ...]:
    return tuple(slice(int(b[0]), int(b[1])) for b in bbox)


def crop_to_nonzero(
    data: np.ndarray,
    reference: np.ndarray | None = None,
    outside_value: int = -1,
) -> tuple[np.ndarray, np.ndarray, List[List[int]]]:
    mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(mask)
    mask = mask[bbox_to_slices(bbox)][None]
    slicer = (slice(None),) + bbox_to_slices(bbox)
    data = data[slicer]
    if reference is None:
        reference = np.where(mask, np.int8(0), np.int8(outside_value))
    else:
        reference = reference[slicer]
        if np.issubdtype(reference.dtype, np.unsignedinteger):
            reference = reference.astype(np.int16, copy=False)
        reference[(reference == 0) & (~mask)] = outside_value
    return data, reference, bbox


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


def resample_array(
    data: np.ndarray,
    new_shape: Sequence[int],
    order: int,
) -> np.ndarray:
    if order < 0:
        _fail_validation(f"Interpolation order must be non-negative, got {order}")
    if len(new_shape) != data.ndim - 1:
        _fail_validation(
            f"new_shape must match data spatial dims, got {len(new_shape)} and data shape {data.shape}"
        )
    if any(i <= 0 for i in new_shape):
        _fail_validation(f"new_shape must contain only positive values, got {tuple(new_shape)}")
    if tuple(data.shape[1:]) == tuple(new_shape):
        return data
    zoom_factors = [1.0] + [float(n) / float(o) for o, n in zip(data.shape[1:], new_shape)]
    return zoom(data, zoom=zoom_factors, order=order)
