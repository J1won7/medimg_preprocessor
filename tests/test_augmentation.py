import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from medimg_preprocessor.augmentation import NNUNetV2Augmentation


def _segmentation_sample():
    image = torch.zeros((1, 8, 12, 12), dtype=torch.float32)
    image[:, 2:6, 3:8, 3:8] = 1.0
    target = torch.zeros((1, 8, 12, 12), dtype=torch.int64)
    target[:, 2:5, 3:6, 3:6] = 1
    target[:, 3:7, 6:10, 6:10] = 2
    mask = target[0] > 0
    return {
        "image": image,
        "target": target,
        "mask": mask,
        "task_mode": "segmentation",
        "reference_type": "segmentation",
    }


def test_disabled_augmentation_preserves_sample():
    sample = _segmentation_sample()
    augmentation = NNUNetV2Augmentation(
        p_rotation=0,
        p_scaling=0,
        p_gaussian_noise=0,
        p_gaussian_blur=0,
        p_brightness=0,
        p_contrast=0,
        p_low_resolution=0,
        p_gamma_invert=0,
        p_gamma=0,
        mirror_axes=(),
    )

    result = augmentation.apply(sample, rng=np.random.RandomState(7))

    assert torch.equal(result["image"], sample["image"])
    assert torch.equal(result["target"], sample["target"])
    assert torch.equal(result["mask"], sample["mask"])


def test_initial_patch_size_reserves_context_for_spatial_transforms():
    augmentation = NNUNetV2Augmentation(
        p_rotation=1,
        p_scaling=1,
        dummy_2d=False,
        p_gaussian_noise=0,
        p_gaussian_blur=0,
        p_brightness=0,
        p_contrast=0,
        p_low_resolution=0,
        p_gamma_invert=0,
        p_gamma=0,
        mirror_axes=(),
    )

    final_patch = (8, 32, 32)
    initial_patch = augmentation.get_initial_patch_size(final_patch)

    assert all(initial >= final for initial, final in zip(initial_patch, final_patch))
    assert initial_patch != final_patch


def test_initial_patch_size_keeps_anisotropic_axis_for_dummy_2d():
    augmentation = NNUNetV2Augmentation(
        p_rotation=1,
        p_scaling=1,
        p_gaussian_noise=0,
        p_gaussian_blur=0,
        p_brightness=0,
        p_contrast=0,
        p_low_resolution=0,
        p_gamma_invert=0,
        p_gamma=0,
        mirror_axes=(),
    )

    final_patch = (8, 32, 32)
    initial_patch = augmentation.get_initial_patch_size(final_patch)

    assert initial_patch[0] == final_patch[0]
    assert initial_patch[1] >= final_patch[1]
    assert initial_patch[2] >= final_patch[2]


def test_augmentation_crops_to_requested_final_patch():
    sample = _segmentation_sample()
    augmentation = NNUNetV2Augmentation(
        p_rotation=0,
        p_scaling=0,
        p_gaussian_noise=0,
        p_gaussian_blur=0,
        p_brightness=0,
        p_contrast=0,
        p_low_resolution=0,
        p_gamma_invert=0,
        p_gamma=0,
        mirror_axes=(),
    )

    result = augmentation.apply(
        sample,
        rng=np.random.RandomState(5),
        crop_size=(4, 8, 8),
    )

    assert result["image"].shape == (1, 4, 8, 8)
    assert result["target"].shape == (1, 4, 8, 8)
    assert result["mask"].shape == (4, 8, 8)


def test_linear_label_and_mask_interpolation_preserve_discrete_values():
    sample = _segmentation_sample()
    augmentation = NNUNetV2Augmentation(
        p_rotation=1,
        p_scaling=1,
        scaling=(0.8, 1.2),
        p_gaussian_noise=0,
        p_gaussian_blur=0,
        p_brightness=0,
        p_contrast=0,
        p_low_resolution=0,
        p_gamma_invert=0,
        p_gamma=0,
        mirror_axes=(),
        label_interpolation="linear",
        mask_interpolation="linear",
    )

    result = augmentation.apply(sample, rng=np.random.RandomState(11))

    assert result["image"].shape == sample["image"].shape
    assert result["target"].dtype == sample["target"].dtype
    assert set(torch.unique(result["target"]).tolist()).issubset({0, 1, 2})
    assert result["mask"].dtype == torch.bool


def test_paired_fields_receive_the_same_spatial_transform():
    image = torch.zeros((1, 8, 12, 12), dtype=torch.float32)
    image[:, 2:6, 3:8, 3:8] = 1.0
    target = image.clone()
    sample = {
        "image": image,
        "target": target,
        "task_mode": "paired_generative",
        "reference_type": "image",
    }
    augmentation = NNUNetV2Augmentation(
        p_rotation=1,
        p_scaling=1,
        p_gaussian_noise=0,
        p_gaussian_blur=0,
        p_brightness=0,
        p_contrast=0,
        p_low_resolution=0,
        p_gamma_invert=0,
        p_gamma=0,
        mirror_axes=(),
    )

    result = augmentation.apply(sample, rng=np.random.RandomState(19))

    assert torch.equal(result["target"], result["image"])
