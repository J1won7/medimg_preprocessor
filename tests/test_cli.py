import pytest


pytest.importorskip("numpy")

from medimg_preprocessor.cli import _assert_matching_identifiers


def test_identifier_matching_returns_intersection_without_raising(capsys):
    matched = _assert_matching_identifiers(
        {"case_a": ["a"], "case_shared": ["shared"], "image_only": ["image"]},
        {"case_shared": ["shared"], "label_only": ["label"]},
        "images",
        "labels",
    )

    assert matched == ["case_shared"]
    output = capsys.readouterr().out
    assert "using 1 matched cases" in output
    assert "ignored 2 images cases" in output
    assert "ignored 1 labels cases" in output
