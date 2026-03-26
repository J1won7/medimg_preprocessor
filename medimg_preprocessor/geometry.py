from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import List, Sequence, Tuple
import warnings

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure, label, map_coordinates
from skimage.transform import resize


ANISO_THRESHOLD = 3.0


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


def postprocess_binary_mask(
    mask: np.ndarray,
    *,
    fill_holes: bool = True,
    keep_largest_component: bool = True,
    closing_iters: int = 1,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if fill_holes:
        mask = binary_fill_holes(mask)
    if closing_iters > 0:
        structure = generate_binary_structure(mask.ndim, 1)
        mask = binary_closing(mask, structure=structure, iterations=int(closing_iters))
    if keep_largest_component and np.any(mask):
        structure = generate_binary_structure(mask.ndim, 1)
        labeled, num = label(mask, structure=structure)
        if num > 1:
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0
            largest = int(np.argmax(component_sizes))
            mask = labeled == largest
    if fill_holes:
        mask = binary_fill_holes(mask)
    return np.asarray(mask, dtype=bool)


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


def get_do_separate_z(spacing: Sequence[float], anisotropy_threshold: float = ANISO_THRESHOLD) -> bool:
    spacing = np.asarray(spacing, dtype=np.float64)
    return bool((np.max(spacing) / np.min(spacing)) > anisotropy_threshold)


def get_lowres_axis(new_spacing: Sequence[float]) -> np.ndarray:
    new_spacing = np.asarray(new_spacing, dtype=np.float64)
    return np.where(max(new_spacing) / new_spacing == 1)[0]


def determine_do_sep_z_and_axis(
    force_separate_z: bool | None,
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
) -> tuple[bool, int | None]:
    if force_separate_z is not None:
        do_separate_z = bool(force_separate_z)
        axis = get_lowres_axis(current_spacing) if do_separate_z else None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
            axis = None
        elif len(axis) == 2:
            do_separate_z = False
            axis = None
        else:
            axis = int(axis[0])
    return do_separate_z, axis


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


def _resample_data_or_seg(
    data: np.ndarray,
    new_shape: Sequence[int],
    *,
    is_seg: bool = False,
    axis: int | None = None,
    order: int = 3,
    do_separate_z: bool = False,
    order_z: int = 0,
    dtype_out=None,
) -> np.ndarray:
    if data.ndim not in (3, 4):
        _fail_validation(f"data must be (C, X, Y) or (C, X, Y, Z), got {data.shape}")
    if len(new_shape) != data.ndim - 1:
        _fail_validation(f"new_shape must match spatial dims, got {new_shape} for data shape {data.shape}")

    resize_fn = _resize_segmentation if is_seg else resize
    kwargs = OrderedDict() if is_seg else {"mode": "edge", "anti_aliasing": False, "preserve_range": True}

    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if dtype_out is None:
        dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    if np.any(shape != new_shape):
        data = data.astype(float, copy=False)
        if data.ndim == 3:
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
            return reshaped_final
        if do_separate_z:
            if axis is None:
                _fail_validation("axis is required when do_separate_z is True")
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                if shape[axis] != new_shape[axis]:
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape
                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final[c] = map_coordinates(
                            reshaped_here,
                            coord_map,
                            order=order_z,
                            mode="nearest",
                        )[None]
                    else:
                        for label in np.unique(reshaped_here):
                            mask = map_coordinates(
                                (reshaped_here == label).astype(float),
                                coord_map,
                                order=order_z,
                                mode="nearest",
                            )
                            reshaped_final[c][np.round(mask) > 0.5] = label
                else:
                    reshaped_final[c] = reshaped_here
        else:
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        return reshaped_final
    return data


def resample_array(
    data: np.ndarray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    *,
    is_seg: bool,
    order: int,
    order_z: int = 0,
    force_separate_z: bool | None = None,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD,
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
    do_separate_z, axis = determine_do_sep_z_and_axis(
        force_separate_z,
        current_spacing,
        new_spacing,
        separate_z_anisotropy_threshold,
    )
    return _resample_data_or_seg(
        data,
        new_shape,
        is_seg=is_seg,
        axis=axis,
        order=order,
        do_separate_z=do_separate_z,
        order_z=order_z,
    )
