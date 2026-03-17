from .cli import medimg_preprocess
from .config import PreprocessingConfig, ResamplingConfig
from .preprocessing import (
    GenerativePreprocessor,
    ModularPreprocessingSettings,
    ModularPreprocessor,
    PreprocessingMode,
    RunStage,
    SegmentationPreprocessor,
    TaskAwarePreprocessor,
    TaskMode,
    TaskPreprocessedCase,
    aggregate_intensity_properties_from_arrays,
    aggregate_intensity_properties_from_image_files,
    compute_intensity_properties_from_image,
)

try:
    from .imageio import (
        BaseReaderWriter,
        NaturalImage2DIO,
        NibabelIO,
        NibabelIOWithReorient,
        SimpleITKIO,
        SimpleITKIOWithReorient,
        Tiff3DIO,
        determine_reader_writer_from_dataset_json,
        determine_reader_writer_from_file_ending,
        read_nifti_images,
        read_nifti_seg,
    )
except ModuleNotFoundError as e:
    if e.name not in {"nibabel", "SimpleITK", "tifffile", "skimage"}:
        raise

try:
    from .dataset import (
        PairedGenerativeDataset,
        SegmentationDataset,
        SelfSupervisedDataset,
        TaskPreprocessedDataset,
        UnpairedGenerativeDataset,
        load_preprocessed_dataset,
        load_preprocessed_case,
        load_preprocessed_dataset_manifest,
        save_preprocessed_dataset,
        save_preprocessed_dataset_manifest,
        save_preprocessed_case,
        save_unpaired_preprocessed_dataset_manifest,
    )
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise

__all__ = [
    "GenerativePreprocessor",
    "ModularPreprocessingSettings",
    "ModularPreprocessor",
    "medimg_preprocess",
    "PreprocessingConfig",
    "ResamplingConfig",
    "PreprocessingMode",
    "RunStage",
    "SegmentationPreprocessor",
    "TaskAwarePreprocessor",
    "TaskMode",
    "TaskPreprocessedCase",
    "aggregate_intensity_properties_from_arrays",
    "aggregate_intensity_properties_from_image_files",
    "compute_intensity_properties_from_image",
]

_imageio_exports = [
    "BaseReaderWriter",
    "NaturalImage2DIO",
    "NibabelIO",
    "NibabelIOWithReorient",
    "SimpleITKIO",
    "SimpleITKIOWithReorient",
    "Tiff3DIO",
    "determine_reader_writer_from_dataset_json",
    "determine_reader_writer_from_file_ending",
    "read_nifti_images",
    "read_nifti_seg",
]

_dataset_exports = [
    "PairedGenerativeDataset",
    "SegmentationDataset",
    "SelfSupervisedDataset",
    "TaskPreprocessedDataset",
    "UnpairedGenerativeDataset",
    "load_preprocessed_dataset",
    "load_preprocessed_case",
    "load_preprocessed_dataset_manifest",
    "save_preprocessed_dataset",
    "save_preprocessed_dataset_manifest",
    "save_preprocessed_case",
    "save_unpaired_preprocessed_dataset_manifest",
]

for _name in _imageio_exports:
    if _name in globals():
        __all__.append(_name)

for _name in _dataset_exports:
    if _name in globals():
        __all__.append(_name)
