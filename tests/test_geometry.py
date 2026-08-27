import numpy as np
import pytest

import medimg_preprocessor.geometry as geometry


def test_binary_mask_postprocess_fills_an_enclosed_hole():
    mask = np.ones((7, 7), dtype=bool)
    mask[3, 3] = False

    result = geometry.postprocess_binary_mask(mask, operation="fill_holes")

    assert result.dtype == np.bool_
    assert result[3, 3]


def test_binary_mask_postprocess_can_keep_only_the_largest_component():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:5, 1:5] = True
    mask[6, 6] = True

    result = geometry.postprocess_binary_mask(
        mask,
        operation="none",
        keep_largest_component=True,
    )

    assert result[2, 2]
    assert not result[6, 6]


def test_label_conflict_uses_nearest_original_instance(monkeypatch):
    label_map = np.zeros((7, 9), dtype=np.int16)
    label_map[3, 1] = 1
    label_map[3, 8] = 2

    def claim_shared_voxel(instance, *, operation, closing_iters):
        result = np.asarray(instance, dtype=bool).copy()
        result[3, 4] = True
        return result

    monkeypatch.setattr(geometry, "_apply_morphology", claim_shared_voxel)

    result = geometry.postprocess_label_instances(
        label_map,
        operation="closing",
        closing_iters=1,
    )

    assert result[3, 4] == 1


def test_label_conflict_tie_uses_smaller_label_id(monkeypatch):
    label_map = np.zeros((7, 7), dtype=np.int16)
    label_map[3, 1] = 2
    label_map[3, 5] = 1

    def claim_shared_voxel(instance, *, operation, closing_iters):
        result = np.asarray(instance, dtype=bool).copy()
        result[3, 3] = True
        return result

    monkeypatch.setattr(geometry, "_apply_morphology", claim_shared_voxel)

    result = geometry.postprocess_label_instances(
        label_map,
        operation="closing",
        closing_iters=1,
    )

    assert result[3, 3] == 1


def test_label_linear_resampling_preserves_the_original_label_set():
    pytest.importorskip("SimpleITK")
    # Label-aware interpolation must not treat instance IDs as numeric
    # intensities or manufacture fractional/unknown IDs.
    source = np.array(
        [
            [1, 2],
            [3, 1],
        ],
        dtype=np.int16,
    )[None]

    result = geometry.resample_array(
        source,
        (5, 5),
        is_seg=True,
        order=1,
    )

    assert set(np.unique(result).tolist()).issubset({1, 2, 3})


def test_mask_axis_orders_are_applied_deterministically():
    pytest.importorskip("SimpleITK")
    source = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=np.int16,
    )[None]

    result = geometry.resample_array(
        source,
        (5, 4),
        is_seg=True,
        orders=(1, 0),
    )

    assert set(np.unique(result).tolist()).issubset({1, 2, 3, 4})
    assert np.all(result != 0)


def test_image_resampling_uses_all_physical_axes_and_preserves_constant_values():
    pytest.importorskip("SimpleITK")
    source = np.stack(
        [
            np.full((2, 3, 4), 7.0, dtype=np.float32),
            np.full((2, 3, 4), -2.0, dtype=np.float32),
        ]
    )

    result = geometry.resample_image(
        source,
        (4, 6, 8),
        current_spacing=(2.0, 3.0, 4.0),
        new_spacing=(1.0, 1.5, 2.0),
        order=3,
    )

    assert result.shape == (2, 4, 6, 8)
    assert np.allclose(result[0], 7.0)
    assert np.allclose(result[1], -2.0)


def test_image_resampling_keeps_package_axis_order():
    pytest.importorskip("SimpleITK")
    source = np.zeros((1, 2, 3, 4), dtype=np.float32)
    for x in range(2):
        for y in range(3):
            for z in range(4):
                source[0, x, y, z] = 100 * x + 10 * y + z

    result = geometry.resample_image(
        source,
        (4, 6, 8),
        current_spacing=(1.0, 1.0, 1.0),
        new_spacing=(0.5, 0.5, 0.5),
        order=0,
    )

    assert result[0, 0, 2, 4] == 12.0
    assert result[0, 2, 4, 6] == 123.0


def test_mask_resampling_preserves_instance_ids_in_3d():
    pytest.importorskip("SimpleITK")
    source = np.zeros((1, 3, 3, 2), dtype=np.int16)
    source[0, 1:, 1:, :] = 7

    result = geometry.resample_mask(
        source,
        (6, 6, 4),
        current_spacing=(2.0, 2.0, 3.0),
        new_spacing=(1.0, 1.0, 1.5),
        order=1,
    )

    assert result.shape == (1, 6, 6, 4)
    assert set(np.unique(result).tolist()).issubset({0, 7})
    assert np.any(result == 7)
