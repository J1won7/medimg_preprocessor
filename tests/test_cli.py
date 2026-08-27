import pytest


pytest.importorskip("numpy")

from medimg_preprocessor.cli import (
    _assert_matching_identifiers,
    _normalize_legacy_resampling_args,
    build_parser,
)


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


def test_legacy_label_interpolation_is_normalized_to_mask_policy():
    args = build_parser().parse_args(
        [
            "preprocess-dataset",
            "--task-mode",
            "segmentation",
            "--images-dir",
            "images",
            "--output-folder",
            "output",
            "--label-interpolation",
            "linear",
        ]
    )

    _normalize_legacy_resampling_args(args)

    assert args.mask_interpolation == "linear"

    numeric_args = build_parser().parse_args(
        [
            "preprocess-dataset",
            "--task-mode",
            "segmentation",
            "--images-dir",
            "images",
            "--output-folder",
            "output",
            "--label-order",
            "1",
        ]
    )
    _normalize_legacy_resampling_args(numeric_args)
    assert numeric_args.mask_order == 1


def test_canonical_and_legacy_mask_interpolation_options_cannot_be_combined():
    args = build_parser().parse_args(
        [
            "preprocess-dataset",
            "--task-mode",
            "segmentation",
            "--images-dir",
            "images",
            "--output-folder",
            "output",
            "--mask-interpolation",
            "nearest",
            "--label-interpolation",
            "linear",
        ]
    )

    with pytest.raises(ValueError, match="mask-interpolation"):
        _normalize_legacy_resampling_args(args)
