import pytest


np = pytest.importorskip("numpy")

from medimg_preprocessor.inference import (
    InferencePatchAccumulator,
    _config_from_manifest_payload,
    _output_starts_for_padded_input,
)


def test_accumulator_discards_padded_patch_border():
    accumulator = InferencePatchAccumulator((93,), channels=1)
    patch = np.full((1, 96), -1.0, dtype=np.float32)
    patch[0, 1:94] = np.arange(93, dtype=np.float32)

    # The patch was centered with one voxel of left padding, so its source
    # voxel 0 maps to original output voxel 0 at start=-1.
    accumulator.add_patch(patch, starts=(-1,))

    result = accumulator.finalize()
    assert result.shape == (1, 93)
    assert np.array_equal(result[0], np.arange(93, dtype=np.float32))


def test_manifest_config_restores_axis_specific_resampling_orders():
    config = _config_from_manifest_payload(
        {
            "spacing": [1.0, 1.0, 1.0],
            "transpose_forward": [0, 1, 2],
            "normalization_schemes": ["ZScoreNormalization"],
            "use_mask_for_norm": [False],
            "foreground_intensity_properties_per_channel": {"0": {"mean": 0.0, "std": 1.0}},
            "resampling": {
                "image_order": 3,
                "image_orders": [3, 1, 3],
                "label_order": 1,
                "label_orders": [1, 1, 1],
                "mask_order": 0,
                "mask_orders": [0, 0, 0],
            },
        }
    )

    assert config.resampling.orders_for("image", 3) == (3, 1, 3)
    assert config.resampling.orders_for("label", 3) == (1, 1, 1)


def test_padded_input_uses_negative_output_start():
    starts = _output_starts_for_padded_input((93, 128), (96, 64), ((0, 0), (0, 64)))

    assert starts == [(-1, 0), (-1, 64)]
