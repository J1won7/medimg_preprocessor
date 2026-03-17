# medimg_preprocessor

`medimg_preprocessor` is a medical image preprocessing library built to keep nnU-Net-style preprocessing behavior while being easier to reuse in other projects.

The goal is simple:

- keep preprocessing close to nnU-Net
- make CLI usage simpler
- make the output easy to plug into PyTorch training
- support segmentation, paired generative, unpaired generative, and self-supervised workflows with one interface

## What It Covers

This package focuses on preprocessing, not full nnU-Net experiment planning for network architecture.

It includes:

- medical image reader support for `nii`, `nii.gz`, `nrrd`, `mha`, `gipl`, `tif`, `tiff`, `png`, `bmp`
- header-based spacing handling
- nonzero cropping
- nnU-Net-style automatic fingerprinting and preprocessing planning
- anisotropy-aware target spacing and resampling
- channel-name-based normalization selection when `dataset.json` is available
- saved preprocessed cases and manifest-based dataset loading

## Install

```bash
pip install git+https://github.com/J1won7/medimg_preprocessor.git
```

Main runtime dependencies are declared in the package:

- `numpy`
- `scipy`
- `torch`
- `nibabel`
- `SimpleITK`
- `tifffile`
- `scikit-image`

## Typical Workflow

```text
raw directories
-> preprocess-dataset
-> saved .npz/.pkl cases + preprocessing_manifest.json
-> load_preprocessed_dataset(...)
-> PyTorch DataLoader
-> training
```

## Quick Start

### Segmentation

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir imagesTr \
  --labels-dir labelsTr \
  --output-folder preprocessed
```

### Paired Generative

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --source-dir source \
  --target-dir target \
  --output-folder preprocessed_paired
```

### Unpaired Generative

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode unpaired_generative \
  --domain-a-dir domain_a \
  --domain-b-dir domain_b \
  --output-folder preprocessed_unpaired
```

### Self-Supervised

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode self_supervised \
  --images-dir images \
  --output-folder preprocessed_ssl
```

## Automatic Behavior

If you do not provide `--config-json` or `--plans-file`, the package will automatically:

- scan the dataset
- read voxel spacing from image headers
- crop to nonzero
- build a fingerprint
- estimate target spacing in an nnU-Net-style way
- decide transpose order
- decide normalization schemes
- collect CT foreground intensity statistics when possible

If a neighboring `dataset.json` exists, it is used for:

- reader selection
- `channel_names` / `modality`
- nnU-Net-style normalization mapping

## Directory Rules

### Segmentation

Use:

- `--images-dir`
- `--labels-dir`

Cases are matched by case identifier.

Examples:

- `imagesTr/case_0001_0000.nii.gz`
- `labelsTr/case_0001.nii.gz`

Single-channel nnU-Net image names like `case_0001_0000.nii.gz` are handled automatically.

### Paired Generative

Use:

- `--source-dir`
- `--target-dir`

Cases are matched by case identifier.

Examples:

- `source/case_0001_0000.nii.gz`
- `target/case_0001_0000.nii.gz`

or

- `source/case_0001.nii.gz`
- `target/case_0001.nii.gz`

### Unpaired Generative

Use:

- `--domain-a-dir`
- `--domain-b-dir`

There is no 1:1 pairing between the two domains at preprocessing time.

### Self-Supervised

Use:

- `--images-dir`

Only image files are scanned.

## Multi-Image Cases

If one case is composed of multiple image files, use:

```bash
--multi-image
```

Then filenames must follow this rule:

- `case_0001_0000.nii.gz`
- `case_0001_0001.nii.gz`
- `case_0001_0002.nii.gz`

Rules:

- postfix must be 4 digits
- numbering must start at `0000`
- numbering must be contiguous

This applies to image folders:

- `--images-dir`
- `--source-dir`
- `--target-dir`
- `--domain-a-dir`
- `--domain-b-dir`

For segmentation labels, `--labels-dir` is still one file per case.

## Overriding Automatic Planning

Automatic planning is the default.

If needed, you can override it with either:

### Direct config JSON

```bash
--config-json config.json
```

### nnU-Net plans

```bash
--plans-file nnUNetPlans.json --configuration-name 3d_fullres
```

For unpaired generative, you may provide:

- one shared config with `--config-json`
- separate configs with `--config-a-json` and `--config-b-json`

## Output Structure

### Single-folder tasks

```text
preprocessed/
  case_0001.npz
  case_0001.pkl
  case_0002.npz
  case_0002.pkl
  preprocessing_manifest.json
```

### Unpaired tasks

```text
preprocessed_unpaired/
  domain_a/
    a_0001.npz
    a_0001.pkl
  domain_b/
    b_0001.npz
    b_0001.pkl
  preprocessing_manifest.json
```

## Loading in PyTorch

```python
from torch.utils.data import DataLoader
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset("preprocessed")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

batch = next(iter(loader))
```

Returned keys depend on the task:

- segmentation: `image`, `target`
- paired generative: `image`, `target`
- unpaired generative: `image_a`, `image_b`
- predict / predict_and_evaluate: `image`, optionally `evaluation_reference`

## CLI

Main commands:

```bash
python -m medimg_preprocessor preprocess-dataset --help
python -m medimg_preprocessor show-manifest --help
python -m medimg_preprocessor save-dataset --help
```

After installation:

```bash
medimg-preprocess --help
```

## Manifest

To inspect a saved dataset:

```bash
python -m medimg_preprocessor show-manifest --folder preprocessed
```

If case files already exist and you only want to write the manifest:

```bash
python -m medimg_preprocessor save-dataset --help
```

## Fail-Fast Policy

This package is intentionally fail-fast.

It warns and raises instead of silently continuing when:

- image/label identifiers do not match
- source/target identifiers do not match
- a multi-image case uses invalid postfix numbering
- shapes or spacings are inconsistent
- config values are invalid
- a required target is missing

## Positioning

This is not meant to be a full replacement for nnU-Net.

It is meant to be:

- very close to nnU-Net preprocessing
- easier to call from custom code
- easier to use for non-segmentation workflows
- easier to load into standard PyTorch training pipelines
