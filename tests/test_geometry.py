import numpy as np

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
