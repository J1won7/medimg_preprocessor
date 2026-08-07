import pytest


np = pytest.importorskip("numpy")

from medimg_preprocessor.dataset import _crop_with_starts_padded


def test_oversized_patch_keeps_requested_shape():
    source = np.arange(93, dtype=np.int16)[None]

    result = _crop_with_starts_padded(source, (96,), (0,))

    assert result.shape == (1, 96)
    assert np.array_equal(result[0, 1:94], source[0])
    assert result[0, 0] == 0
    assert result[0, 94] == 0
    assert result[0, 95] == 0


def test_negative_initial_start_keeps_requested_shape():
    source = np.ones((1, 93), dtype=np.int16)

    result = _crop_with_starts_padded(source, (96,), (-1,))

    assert result.shape == (1, 96)
