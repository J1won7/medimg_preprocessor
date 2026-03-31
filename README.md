# medimg_preprocessor

Medical image preprocessing utility for:

- `segmentation`
- `paired_generative`
- `unpaired_generative`
- `self_supervised`

It provides:

- dataset planning
- preprocessing
- saved preprocessed datasets
- PyTorch dataset loading

## Installation

```bash
pip install -e .
```

Or install from Git:

```bash
pip install git+https://github.com/J1won7/medimg_preprocessor.git
```

## Preprocessing

### Basic command

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode <mode> \
  --images-dir <images_dir> \
  --output-folder <output_folder>
```

Add `--target-dir` when the selected mode needs a target.

### Task modes

- `segmentation`
  - input: `--images-dir`
  - target: `--target-dir`
  - uses label-derived mask automatically

- `paired_generative`
  - input: `--images-dir`
  - target: `--target-dir`
  - supports image-side and target-side masks

- `unpaired_generative`
  - input: `--images-dir`
  - target: `--target-dir`
  - domain A and domain B are preprocessed separately

- `self_supervised`
  - input: `--images-dir`
  - no target directory

### Quick examples

#### 1. Segmentation

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir raw/imagesTr \
  --target-dir raw/labelsTr \
  --output-folder preprocessed_seg
```

#### 2. Paired Generative

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --images-dir raw/source \
  --target-dir raw/target \
  --output-folder preprocessed_paired
```

#### 3. Unpaired Generative

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode unpaired_generative \
  --images-dir raw/domain_a \
  --target-dir raw/domain_b \
  --output-folder preprocessed_unpaired
```

#### 4. Self-Supervised

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode self_supervised \
  --images-dir raw/images \
  --output-folder preprocessed_ssl
```

### Common parameters

- `--task-mode`
  - preprocessing mode
  - one of `segmentation`, `paired_generative`, `unpaired_generative`, `self_supervised`

- `--images-dir`
  - main input directory

- `--target-dir`
  - target directory
  - required for `segmentation`, `paired_generative`, `unpaired_generative`

- `--output-folder`
  - folder where preprocessed files and manifest are saved

- `--num-processes`
  - number of worker processes used during preprocessing

- `--run-stage`
  - usually `train`
  - controls whether train/val splits and training metadata are created

- `--multi-image`
  - enables multi-channel file grouping such as `case_0001_0000.nii.gz`, `case_0001_0001.nii.gz`

- `--storage-format`
  - output array format
  - usually `blosc2` or `npz`

### Cropping behavior

The preprocessor no longer applies nnU-Net-style nonzero cropping during preprocessing.

- image and target arrays keep their original spatial extent until resampling
- segmentation labels are not rewritten to `-1` outside an image-derived crop box
- saved label semantics stay closer to the original dataset

### Masking

Patch sampling is based on the saved mask when a mask is available.

Default behavior:

- `segmentation`
  - uses the label as the sampling mask automatically

- other modes
  - if no mask is given, patches are sampled from the full spatial range

#### External mask directories

- `--images-mask-dir`
  - mask directory aligned with `--images-dir`

- `--target-mask-dir`
  - mask directory aligned with `--target-dir`

If mask directories are given, they are used first.

Example:

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --images-dir raw/source \
  --target-dir raw/target \
  --images-mask-dir raw/source_masks \
  --target-mask-dir raw/target_masks \
  --output-folder preprocessed_paired
```

#### Threshold-based masks

Use threshold-based masks only when you do not have an external mask.

- `--masking-mode threshold`
  - enables threshold mask generation

- `--mask-threshold`
  - shorthand threshold applied to both image and target sides

- `--images-mask-threshold`
  - threshold only for image-side mask generation

- `--target-mask-threshold`
  - threshold only for target-side mask generation

Rules:

- if only `--mask-threshold` is given, both sides use that threshold
- if `--images-mask-threshold` or `--target-mask-threshold` is also given, that side overrides the common threshold
- use `none` to disable one side explicitly

Examples:

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --images-dir raw/source \
  --target-dir raw/target \
  --masking-mode threshold \
  --mask-threshold -0.8 \
  --output-folder preprocessed_paired
```

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --images-dir raw/source \
  --target-dir raw/target \
  --masking-mode threshold \
  --images-mask-threshold -0.8 \
  --target-mask-threshold none \
  --output-folder preprocessed_paired
```

#### Mask post-processing

- `--mask-fill-holes`
  - fill holes in the generated mask
  - default: enabled

- `--mask-keep-largest-component`
  - keep only the largest connected component
  - default: enabled

- `--mask-closing-iters`
  - binary closing iterations
  - default: `1`

### Patch sampling

The loader now uses nnUNet-style dynamic patch sampling:

- preprocessing saves mask voxel locations
- dataset loading samples a valid voxel from the saved locations
- a patch bbox is created dynamically at runtime

Related options:

- `--patch-mask-max-starts`
  - maximum number of saved foreground/mask voxel locations
  - default: `8192`

- `--patch-mask-min-fraction`
  - legacy fallback option for older precomputed-start workflows
  - usually can be left as default

### Planning and configuration

You can let the tool plan preprocessing automatically, or provide a config/plans file.

- `--config-json`
  - use an explicit preprocessing config

- `--plans-file`
  - load preprocessing settings from nnUNet-style plans

- `--configuration-name`
  - select a configuration inside the plans file

### Label resampling override

If you want to keep automatic planning or nnU-Net plans but force a safer label interpolation mode,
you can override the segmentation label resampling order directly from the CLI.

- `--label-order 0`
  - nearest-neighbor style label resampling
  - safest option when label IDs must be preserved exactly
  - usually preferred for instance IDs or very small structures

- `--label-order 1`
  - resizes each label mask separately and then reconstructs the label map
  - can make boundaries a bit smoother
  - may alter thin structures or tightly packed instances

- `--label-order-z 0`
  - same idea, but only for the separate-z pass used on anisotropic volumes

### Stored mask files

The saved `_mask` file is a stored sampling mask. It is not the same thing as
`use_mask_for_norm` in the normalization config.

- `--save-mask`
  - force writing the derived sampling mask to disk

- `--no-save-mask`
  - skip writing the `_mask` file

Default behavior:

- `segmentation` with `train`
  - `_mask` file is not written by default
  - patch sampling still uses label-derived foreground locations

- other modes or stages
  - `_mask` file is written by default, as before

Example:

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir raw/imagesTr \
  --target-dir raw/labelsTr \
  --output-folder preprocessed_seg \
  --label-order 0 \
  --no-save-mask
```

### Normalization

Useful normalization options:

- `--normalization-method auto`
- `--normalization-method CTNormalization`
- `--normalization-method ZScoreNormalization`
- `--normalization-method MinMaxClipNormalization`

Examples:

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --images-dir raw/source \
  --target-dir raw/target \
  --output-folder preprocessed_paired \
  --normalization-method ZScoreNormalization
```

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --images-dir raw/source \
  --target-dir raw/target \
  --output-folder preprocessed_paired \
  --normalization-method MinMaxClipNormalization \
  --normalization-min -1000 \
  --normalization-max 2000
```

## Dataset Usage

### Basic loading

```python
from torch.utils.data import DataLoader
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset("preprocessed_paired")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

batch = next(iter(loader))
print(batch["image"].shape)
```

### Load a specific configuration

```python
from medimg_preprocessor import load_preprocessed_dataset

dataset_2d = load_preprocessed_dataset("preprocessed_seg", configuration="2d")
dataset_3d = load_preprocessed_dataset("preprocessed_seg", configuration="3d")
```

### Load train/val split

```python
from medimg_preprocessor import load_preprocessed_dataset

train_ds = load_preprocessed_dataset("preprocessed_seg", split="train")
val_ds = load_preprocessed_dataset("preprocessed_seg", split="val")
```

### Self-supervised usage

```python
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset("preprocessed_ssl")
sample = dataset[0]

print(sample["image"].shape)
print(sample["view1"].shape)
print(sample["view2"].shape)
```

### Unpaired usage

```python
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset("preprocessed_unpaired")
sample = dataset[0]

print(sample["image_a"].shape)
print(sample["image_b"].shape)
```

## Inference Usage

### Raw-image inference

Use `RawInferencePatchDataset` when you want to:

- read raw NIfTI or similar image files directly
- preprocess them at runtime
- split large volumes into overlapping patches
- merge patch predictions back into a full volume
- restore the output to the original image space
- save the result with the original NIfTI affine and header

### Important behavior

- `patch_size` is required
- large volumes are processed with sliding-window overlap
- overlapping patch predictions are averaged
- final output is restored to the original spatial shape
- `save_prediction_nifti(...)` keeps the original NIfTI affine/header

### Basic inference example

```python
from torch.utils.data import DataLoader
from medimg_preprocessor import PreprocessingConfig, RawInferencePatchDataset

config = PreprocessingConfig()

dataset = RawInferencePatchDataset(
    images_dir="raw_cbct",
    config=config,
    patch_size=(32, 192, 192),
    overlap=0.5,
    image_reader="auto",
    multi_image=False,
)

loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
```

### Patch prediction and reconstruction

```python
acc = dataset.build_accumulator(case_index=0, channels=1)

for batch in loader:
    pred = model(batch["image"])  # [B, C, D, H, W]
    pred = pred.detach().cpu().numpy()
    starts = batch["starts"].cpu().numpy()

    for i in range(pred.shape[0]):
        acc.add_patch(pred[i], starts[i])

full_pred = acc.finalize()
dataset.save_prediction_nifti(full_pred, case_index=0, output_path="pred_case001.nii.gz")
```

### Returned fields

Each sample from `RawInferencePatchDataset` contains:

- `image`
- `identifier`
- `case_index`
- `patch_index`
- `starts`
- `patch_size`

### Main parameters

- `images_dir`
  - raw image directory

- `config`
  - preprocessing config used at runtime

- `patch_size`
  - required patch size for inference

- `overlap`
  - sliding-window overlap ratio

- `image_reader`
  - image IO backend

- `multi_image`
  - enable grouped multi-channel inputs

## Dataset Classes

### `SegmentationDataset`

- mode: `segmentation`
- target required: yes
- typical use:

```python
from medimg_preprocessor.dataset import SegmentationDataset

dataset = SegmentationDataset("preprocessed_seg", patch_size=(64, 128, 128))
```

### `PairedGenerativeDataset`

- mode: `paired_generative`
- target required: yes
- typical use:

```python
from medimg_preprocessor.dataset import PairedGenerativeDataset

dataset = PairedGenerativeDataset("preprocessed_paired", patch_size=(32, 192, 192))
```

### `SelfSupervisedDataset`

- mode: `self_supervised`
- target required: no
- returns `image`, `view1`, `view2`
- typical use:

```python
from medimg_preprocessor.dataset import SelfSupervisedDataset

dataset = SelfSupervisedDataset("preprocessed_ssl", patch_size=(32, 192, 192))
```

### `UnpairedGenerativeDataset`

- mode: `unpaired_generative`
- target required: no paired target
- returns `image_a`, `image_b`
- typical use:

```python
from medimg_preprocessor.dataset import UnpairedGenerativeDataset

dataset = UnpairedGenerativeDataset(
    folder_a="preprocessed_unpaired/domain_a",
    folder_b="preprocessed_unpaired/domain_b",
    patch_size=(32, 192, 192),
)
```

### `load_preprocessed_dataset`

For most users, this is the recommended entry point.

- reads `preprocessing_manifest.json`
- selects the right dataset class automatically
- applies configuration and split settings automatically

```python
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset("preprocessed_paired")
```

### `RawInferencePatchDataset`

- purpose: runtime preprocessing and patch-wise inference on raw images
- use this for direct inference from NIfTI or similar files
- requires explicit `patch_size`

```python
from medimg_preprocessor import PreprocessingConfig, RawInferencePatchDataset

config = PreprocessingConfig()
dataset = RawInferencePatchDataset(
    images_dir="raw_cbct",
    config=config,
    patch_size=(32, 192, 192),
    overlap=0.5,
)
```

### `InferencePatchAccumulator`

- purpose: merge overlapping patch predictions into one preprocessed-volume prediction
- overlapping regions are averaged
- usually used with `RawInferencePatchDataset.build_accumulator(...)`

```python
acc = dataset.build_accumulator(case_index=0, channels=1)
acc.add_patch(pred_patch, starts)
full_pred = acc.finalize()
```
