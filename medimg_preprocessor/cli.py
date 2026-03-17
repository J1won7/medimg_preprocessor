from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import PreprocessingConfig, ResamplingConfig
from .dataset import (
    load_preprocessed_dataset_manifest,
    save_preprocessed_case,
    save_preprocessed_dataset,
)
from .preprocessing import RunStage, TaskMode


DESCRIPTION = """Manage medimg_preprocessor datasets and manifests.

This CLI supports two workflows:

1. preprocess-dataset
   Read raw files from a JSON spec, save preprocessed .npz/.pkl cases,
   and write preprocessing_manifest.json.

2. save-dataset
   Write preprocessing_manifest.json for a dataset whose preprocessed case
   files already exist on disk.
"""


PREPROCESS_DATASET_EPILOG = """JSON spec examples:
  Single-folder segmentation:
    {
      "cases": [
        {
          "identifier": "case_0001",
          "image_files": ["raw/imagesTr/case_0001_0000.nii.gz"],
          "reference_files": "raw/labelsTr/case_0001.nii.gz"
        }
      ]
    }

  Unpaired generative:
    {
      "domains": {
        "a": [
          {"identifier": "a_0001", "image_files": ["raw/domain_a/a_0001.nii.gz"]}
        ],
        "b": [
          {"identifier": "b_0001", "image_files": ["raw/domain_b/b_0001.nii.gz"]}
        ]
      }
    }

Examples:
  Segmentation preprocessing:
    python -m medimg_preprocessor preprocess-dataset --spec segmentation_spec.json --output-folder preprocessed_seg --task-mode segmentation --run-stage train --config-json config.json

  Unpaired generative preprocessing:
    python -m medimg_preprocessor preprocess-dataset --spec unpaired_spec.json --output-folder preprocessed_unpaired --task-mode unpaired_generative --config-a-json config_a.json --config-b-json config_b.json
"""


SAVE_DATASET_EPILOG = """Examples:
  Single-folder segmentation dataset:
    python -m medimg_preprocessor save-dataset --folder preprocessed_seg --task-mode segmentation --run-stage train --plans-file nnUNetPlans.json --configuration-name 3d_fullres --default-patch-size 96 96 96

  Single-folder paired generative dataset:
    python -m medimg_preprocessor save-dataset --folder preprocessed_paired --task-mode paired_generative --run-stage train --config-json config.json

  Unpaired generative dataset:
    python -m medimg_preprocessor save-dataset --folder preprocessed_unpaired --task-mode unpaired_generative --folder-a domain_a --folder-b domain_b --config-a-json config_a.json --config-b-json config_b.json
"""


SHOW_MANIFEST_EPILOG = """Example:
  python -m medimg_preprocessor show-manifest --folder preprocessed_seg
"""


READER_CHOICES = (
    "auto",
    "nibabel",
    "nibabel_reorient",
    "simpleitk",
    "simpleitk_reorient",
    "tiff3d",
    "natural_2d",
)


def _load_config_from_json(path: Optional[str]) -> Optional[PreprocessingConfig]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config JSON must contain an object, got {type(payload).__name__}")
    resampling_payload = payload.get("resampling", {})
    if not isinstance(resampling_payload, dict):
        raise ValueError("Config JSON field 'resampling' must be an object when provided")
    return PreprocessingConfig(
        spacing=payload["spacing"],
        transpose_forward=payload["transpose_forward"],
        normalization_schemes=payload["normalization_schemes"],
        use_mask_for_norm=payload["use_mask_for_norm"],
        foreground_intensity_properties_per_channel=payload.get(
            "foreground_intensity_properties_per_channel",
            {},
        ),
        resampling=ResamplingConfig(
            image_order=int(resampling_payload.get("image_order", 3)),
            label_order=int(resampling_payload.get("label_order", 0)),
        ),
    )


def _load_config(
    config_json: Optional[str],
    plans_file: Optional[str],
    configuration_name: Optional[str],
) -> Optional[PreprocessingConfig]:
    if config_json is not None and plans_file is not None:
        raise ValueError("Use either --config-json or --plans-file, not both")
    if config_json is not None:
        return _load_config_from_json(config_json)
    if plans_file is not None:
        if configuration_name is None:
            raise ValueError("--configuration-name is required when using --plans-file")
        return PreprocessingConfig.from_nnunet_plans(plans_file, configuration_name)
    if configuration_name is not None:
        raise ValueError("--configuration-name requires --plans-file")
    return None


def _load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Spec JSON must contain an object, got {type(payload).__name__}")
    return payload


def _normalize_image_files(value, *, field_name: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or len(value) == 0 or any(not isinstance(i, str) for i in value):
        raise ValueError(f"{field_name} must be a non-empty string list or a string")
    return value


def _normalize_reference_files(value, *, segmentation_reference: bool):
    if value is None:
        return None
    if segmentation_reference and isinstance(value, str):
        return value
    return _normalize_image_files(value, field_name="reference_files")


def _validate_case_entry(case_payload: dict, *, context: str) -> tuple[str, list[str], object]:
    if not isinstance(case_payload, dict):
        raise ValueError(f"{context} must be an object")
    identifier = case_payload.get("identifier")
    if not isinstance(identifier, str) or len(identifier.strip()) == 0:
        raise ValueError(f"{context}.identifier must be a non-empty string")
    image_files = _normalize_image_files(case_payload.get("image_files"), field_name=f"{context}.image_files")
    return identifier, image_files, case_payload.get("reference_files")


def _detect_file_ending(filename: str) -> str:
    lower = filename.lower()
    for ending in (".nii.gz", ".nii", ".nrrd", ".mha", ".gipl", ".tiff", ".tif", ".png", ".bmp"):
        if lower.endswith(ending):
            return ending
    return Path(filename).suffix.lower()


def _build_reader(reader_name: str, example_file: str):
    from .imageio import (
        NaturalImage2DIO,
        NibabelIO,
        NibabelIOWithReorient,
        SimpleITKIO,
        SimpleITKIOWithReorient,
        Tiff3DIO,
        determine_reader_writer_from_file_ending,
    )

    registry = {
        "nibabel": NibabelIO,
        "nibabel_reorient": NibabelIOWithReorient,
        "simpleitk": SimpleITKIO,
        "simpleitk_reorient": SimpleITKIOWithReorient,
        "tiff3d": Tiff3DIO,
        "natural_2d": NaturalImage2DIO,
    }
    if reader_name == "auto":
        ending = _detect_file_ending(example_file)
        return determine_reader_writer_from_file_ending(ending, example_file=example_file, verbose=False)()
    if reader_name not in registry:
        raise ValueError(f"Unsupported reader '{reader_name}'")
    return registry[reader_name]()


def _prepare_output_prefix(output_folder: Path, identifier: str) -> Path:
    output_prefix = output_folder / identifier
    if output_prefix.with_suffix(".npz").exists() or output_prefix.with_suffix(".pkl").exists():
        raise ValueError(f"Refusing to overwrite existing preprocessed case '{identifier}' in {output_folder}")
    return output_prefix


def _preprocess_single_folder_dataset(
    *,
    spec: dict,
    output_folder: Path,
    task_mode: str,
    run_stage: str,
    config: PreprocessingConfig,
    default_patch_size,
    image_reader_name: str,
    reference_reader_name: Optional[str],
) -> str:
    from .preprocessing import TaskAwarePreprocessor

    cases = spec.get("cases")
    if not isinstance(cases, list) or len(cases) == 0:
        raise ValueError("Spec for single-folder datasets must contain a non-empty 'cases' list")
    output_folder.mkdir(parents=True, exist_ok=True)
    preprocessor = TaskAwarePreprocessor(config)
    seen_identifiers: list[str] = []
    seen_identifier_set: set[str] = set()
    segmentation_reference = task_mode == TaskMode.SEGMENTATION

    for index, case_payload in enumerate(cases):
        context = f"cases[{index}]"
        identifier, image_files, reference_files = _validate_case_entry(case_payload, context=context)
        if identifier in seen_identifier_set:
            raise ValueError(f"Duplicate identifier '{identifier}' in spec")
        seen_identifiers.append(identifier)
        seen_identifier_set.add(identifier)
        if run_stage in {RunStage.TRAIN, RunStage.PREDICT_AND_EVALUATE} and task_mode in {
            TaskMode.SEGMENTATION,
            TaskMode.PAIRED_GENERATIVE,
        } and reference_files is None:
            raise ValueError(f"{context}.reference_files is required for {task_mode} {run_stage}")
        normalized_reference_files = _normalize_reference_files(
            reference_files,
            segmentation_reference=segmentation_reference,
        )
        case_image_reader = _build_reader(case_payload.get("image_reader", image_reader_name), image_files[0])
        case_reference_reader = None
        if normalized_reference_files is not None:
            reference_example = (
                normalized_reference_files
                if isinstance(normalized_reference_files, str)
                else normalized_reference_files[0]
            )
            case_reference_reader = _build_reader(
                case_payload.get("reference_reader", reference_reader_name or image_reader_name),
                reference_example,
            )
        case = preprocessor.run_task_case_from_files(
            image_files=image_files,
            image_reader=case_image_reader,
            task_mode=task_mode,
            run_stage=run_stage,
            reference_files=normalized_reference_files,
            reference_reader=case_reference_reader,
        )
        save_preprocessed_case(case, str(_prepare_output_prefix(output_folder, identifier)))

    return save_preprocessed_dataset(
        folder=str(output_folder),
        task_mode=task_mode,
        run_stage=run_stage,
        config=config,
        default_patch_size=default_patch_size,
        identifiers=seen_identifiers,
    )


def _preprocess_unpaired_dataset(
    *,
    spec: dict,
    output_folder: Path,
    run_stage: str,
    config_a: PreprocessingConfig,
    config_b: PreprocessingConfig,
    default_patch_size,
    image_reader_a_name: str,
    image_reader_b_name: str,
    folder_a_name: str,
    folder_b_name: str,
) -> str:
    from .preprocessing import TaskAwarePreprocessor

    domains = spec.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("Spec for unpaired datasets must contain a 'domains' object")
    cases_a = domains.get("a")
    cases_b = domains.get("b")
    if not isinstance(cases_a, list) or len(cases_a) == 0:
        raise ValueError("Spec domain 'a' must be a non-empty list")
    if not isinstance(cases_b, list) or len(cases_b) == 0:
        raise ValueError("Spec domain 'b' must be a non-empty list")

    output_folder.mkdir(parents=True, exist_ok=True)
    folder_a = output_folder / folder_a_name
    folder_b = output_folder / folder_b_name
    folder_a.mkdir(exist_ok=True)
    folder_b.mkdir(exist_ok=True)

    domain_specs = (
        ("a", cases_a, folder_a, config_a, image_reader_a_name),
        ("b", cases_b, folder_b, config_b, image_reader_b_name),
    )
    domain_identifiers: dict[str, list[str]] = {"a": [], "b": []}
    domain_identifier_sets: dict[str, set[str]] = {"a": set(), "b": set()}

    for domain_name, cases, folder, config, reader_name in domain_specs:
        preprocessor = TaskAwarePreprocessor(config)
        for index, case_payload in enumerate(cases):
            context = f"domains.{domain_name}[{index}]"
            identifier, image_files, reference_files = _validate_case_entry(case_payload, context=context)
            if reference_files is not None:
                raise ValueError(f"{context}.reference_files is not allowed for unpaired_generative")
            if identifier in domain_identifier_sets[domain_name]:
                raise ValueError(f"Duplicate identifier '{identifier}' in domain '{domain_name}'")
            domain_identifier_sets[domain_name].add(identifier)
            domain_identifiers[domain_name].append(identifier)
            case_image_reader = _build_reader(case_payload.get("image_reader", reader_name), image_files[0])
            case = preprocessor.run_task_case_from_files(
                image_files=image_files,
                image_reader=case_image_reader,
                task_mode=TaskMode.UNPAIRED_GENERATIVE,
                run_stage=run_stage,
            )
            save_preprocessed_case(case, str(_prepare_output_prefix(folder, identifier)))

    return save_preprocessed_dataset(
        folder=str(output_folder),
        task_mode=TaskMode.UNPAIRED_GENERATIVE,
        run_stage=run_stage,
        default_patch_size=default_patch_size,
        folder_a=folder_a_name,
        folder_b=folder_b_name,
        config_a=config_a,
        config_b=config_b,
        identifiers_a=domain_identifiers["a"],
        identifiers_b=domain_identifiers["b"],
    )


def _preprocess_dataset_command(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    base_config = _load_config(args.config_json, args.plans_file, args.configuration_name)
    config_a = _load_config(args.config_a_json, args.plans_a_file, args.configuration_a_name)
    config_b = _load_config(args.config_b_json, args.plans_b_file, args.configuration_b_name)
    output_folder = Path(args.output_folder)

    if args.task_mode == TaskMode.UNPAIRED_GENERATIVE:
        config_a = config_a or base_config
        config_b = config_b or base_config
        if config_a is None or config_b is None:
            raise ValueError(
                "preprocess-dataset for unpaired_generative requires config for both domains. "
                "Use --config-a-json/--config-b-json or provide a shared config with --config-json."
            )
        manifest_file = _preprocess_unpaired_dataset(
            spec=spec,
            output_folder=output_folder,
            run_stage=args.run_stage,
            config_a=config_a,
            config_b=config_b,
            default_patch_size=args.default_patch_size,
            image_reader_a_name=args.image_reader_a,
            image_reader_b_name=args.image_reader_b,
            folder_a_name=args.folder_a_name,
            folder_b_name=args.folder_b_name,
        )
    else:
        if base_config is None:
            raise ValueError(
                "preprocess-dataset requires --config-json or --plans-file/--configuration-name "
                "for single-folder datasets."
            )
        manifest_file = _preprocess_single_folder_dataset(
            spec=spec,
            output_folder=output_folder,
            task_mode=args.task_mode,
            run_stage=args.run_stage,
            config=base_config,
            default_patch_size=args.default_patch_size,
            image_reader_name=args.image_reader,
            reference_reader_name=args.reference_reader,
        )

    print(Path(manifest_file).resolve())
    return 0


def _save_dataset_command(args: argparse.Namespace) -> int:
    config = _load_config(args.config_json, args.plans_file, args.configuration_name)
    config_a = _load_config(args.config_a_json, args.plans_a_file, args.configuration_a_name)
    config_b = _load_config(args.config_b_json, args.plans_b_file, args.configuration_b_name)

    manifest_file = save_preprocessed_dataset(
        folder=args.folder,
        task_mode=args.task_mode,
        run_stage=args.run_stage,
        config=config,
        default_patch_size=args.default_patch_size,
        folder_a=args.folder_a,
        folder_b=args.folder_b,
        config_a=config_a,
        config_b=config_b,
        random_pairing=not args.disable_random_pairing,
    )
    print(Path(manifest_file).resolve())
    return 0


def _show_manifest_command(args: argparse.Namespace) -> int:
    manifest = load_preprocessed_dataset_manifest(args.folder)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _build_config_argument_group(
    parser: argparse.ArgumentParser,
    *,
    title: str,
    json_flag: str,
    plans_flag: str,
    configuration_flag: str,
    label: str,
) -> argparse._ArgumentGroup:
    group = parser.add_argument_group(title)
    group.add_argument(
        json_flag,
        default=None,
        help=f"Path to a JSON file that directly defines PreprocessingConfig for {label}.",
    )
    group.add_argument(
        plans_flag,
        default=None,
        help=f"Path to an nnU-Net plans JSON file for {label}.",
    )
    group.add_argument(
        configuration_flag,
        default=None,
        help=f"nnU-Net configuration name used with {plans_flag} for {label}.",
    )
    return group


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m medimg_preprocessor",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess-dataset",
        help="Read raw files, save preprocessed cases, and write preprocessing_manifest.json",
        description=(
            "Read raw image files from a JSON spec, preprocess each case, save .npz/.pkl case files, "
            "and write preprocessing_manifest.json."
        ),
        epilog=PREPROCESS_DATASET_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    preprocess_parser.add_argument(
        "--spec",
        required=True,
        help="Path to a JSON file describing the cases to preprocess.",
    )
    preprocess_parser.add_argument(
        "--output-folder",
        required=True,
        help="Output folder for saved preprocessed cases and preprocessing_manifest.json.",
    )
    preprocess_parser.add_argument(
        "--task-mode",
        required=True,
        choices=(
            TaskMode.SEGMENTATION,
            TaskMode.PAIRED_GENERATIVE,
            TaskMode.UNPAIRED_GENERATIVE,
            TaskMode.SELF_SUPERVISED,
        ),
        help="Task mode used for preprocessing.",
    )
    preprocess_parser.add_argument(
        "--run-stage",
        default=RunStage.TRAIN,
        choices=(RunStage.TRAIN, RunStage.PREDICT, RunStage.PREDICT_AND_EVALUATE),
        help="Run stage used for preprocessing. Default: train.",
    )
    preprocess_parser.add_argument(
        "--default-patch-size",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help="Default patch size written into the generated manifest.",
    )
    preprocess_parser.add_argument(
        "--image-reader",
        default="auto",
        choices=READER_CHOICES,
        help="Reader for single-folder dataset images. Default: auto.",
    )
    preprocess_parser.add_argument(
        "--reference-reader",
        default=None,
        choices=READER_CHOICES,
        help="Reader for single-folder dataset references. Default: use the image reader.",
    )
    preprocess_domain_group = preprocess_parser.add_argument_group("unpaired dataset options")
    preprocess_domain_group.add_argument(
        "--image-reader-a",
        default="auto",
        choices=READER_CHOICES,
        help="Reader for domain A images in unpaired_generative. Default: auto.",
    )
    preprocess_domain_group.add_argument(
        "--image-reader-b",
        default="auto",
        choices=READER_CHOICES,
        help="Reader for domain B images in unpaired_generative. Default: auto.",
    )
    preprocess_domain_group.add_argument(
        "--folder-a-name",
        default="domain_a",
        help="Subfolder name created under --output-folder for unpaired domain A.",
    )
    preprocess_domain_group.add_argument(
        "--folder-b-name",
        default="domain_b",
        help="Subfolder name created under --output-folder for unpaired domain B.",
    )
    _build_config_argument_group(
        preprocess_parser,
        title="single-folder dataset config",
        json_flag="--config-json",
        plans_flag="--plans-file",
        configuration_flag="--configuration-name",
        label="the single-folder dataset",
    )
    _build_config_argument_group(
        preprocess_parser,
        title="unpaired domain A config",
        json_flag="--config-a-json",
        plans_flag="--plans-a-file",
        configuration_flag="--configuration-a-name",
        label="domain A",
    )
    _build_config_argument_group(
        preprocess_parser,
        title="unpaired domain B config",
        json_flag="--config-b-json",
        plans_flag="--plans-b-file",
        configuration_flag="--configuration-b-name",
        label="domain B",
    )
    preprocess_parser.set_defaults(func=_preprocess_dataset_command)

    save_parser = subparsers.add_parser(
        "save-dataset",
        help="Write preprocessing_manifest.json for an already-preprocessed dataset",
        description=(
            "Write preprocessing_manifest.json for a directory that already contains "
            "saved preprocessed cases (.npz/.pkl)."
        ),
        epilog=SAVE_DATASET_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    save_parser.add_argument(
        "--folder",
        required=True,
        help=(
            "Dataset folder. For single-folder datasets this is the folder containing the case files. "
            "For unpaired datasets this is the root folder that will contain preprocessing_manifest.json."
        ),
    )
    save_parser.add_argument(
        "--task-mode",
        required=True,
        choices=(
            TaskMode.SEGMENTATION,
            TaskMode.PAIRED_GENERATIVE,
            TaskMode.UNPAIRED_GENERATIVE,
            TaskMode.SELF_SUPERVISED,
        ),
        help="Task type encoded in the manifest.",
    )
    save_parser.add_argument(
        "--run-stage",
        default=RunStage.TRAIN,
        choices=(RunStage.TRAIN, RunStage.PREDICT, RunStage.PREDICT_AND_EVALUATE),
        help="Run stage encoded in the manifest. Default: train.",
    )
    save_parser.add_argument(
        "--default-patch-size",
        type=int,
        nargs="+",
        default=None,
        metavar="DIM",
        help="Default patch size stored in the manifest, for example: --default-patch-size 96 96 96",
    )
    domain_group = save_parser.add_argument_group("unpaired dataset options")
    domain_group.add_argument(
        "--folder-a",
        default=None,
        help="Relative or absolute domain A folder. Required only for task_mode=unpaired_generative.",
    )
    domain_group.add_argument(
        "--folder-b",
        default=None,
        help="Relative or absolute domain B folder. Required only for task_mode=unpaired_generative.",
    )
    domain_group.add_argument(
        "--disable-random-pairing",
        action="store_true",
        help="Store random_pairing=false in the unpaired manifest.",
    )
    _build_config_argument_group(
        save_parser,
        title="single-folder dataset config",
        json_flag="--config-json",
        plans_flag="--plans-file",
        configuration_flag="--configuration-name",
        label="the single-folder dataset",
    )
    _build_config_argument_group(
        save_parser,
        title="unpaired domain A config",
        json_flag="--config-a-json",
        plans_flag="--plans-a-file",
        configuration_flag="--configuration-a-name",
        label="domain A",
    )
    _build_config_argument_group(
        save_parser,
        title="unpaired domain B config",
        json_flag="--config-b-json",
        plans_flag="--plans-b-file",
        configuration_flag="--configuration-b-name",
        label="domain B",
    )
    save_parser.set_defaults(func=_save_dataset_command)

    show_parser = subparsers.add_parser(
        "show-manifest",
        help="Print preprocessing_manifest.json as formatted JSON",
        description="Load preprocessing_manifest.json from a preprocessed dataset folder and print it.",
        epilog=SHOW_MANIFEST_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    show_parser.add_argument(
        "--folder",
        required=True,
        help="Dataset folder containing preprocessing_manifest.json",
    )
    show_parser.set_defaults(func=_show_manifest_command)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def medimg_preprocess(argv: Optional[Sequence[str]] = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
