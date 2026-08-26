# medimg_preprocessor

CT, MRI, CBCT 등 의료 영상을 위한 범용 전처리 도구입니다. nnU-Net의 planning
아이디어를 참고하지만 특정 학습 프레임워크에 종속되지 않으며, segmentation,
instance segmentation, paired/unpaired generative, self-supervised 작업을 지원합니다.

## 주요 기능

- 데이터셋 자동 스캔과 case 매칭
- voxel spacing 변경과 영상/label/mask 보간
- CT/MRI에 사용할 normalization 설정
- 외부 mask 또는 threshold 기반 patch sampling
- semantic/instance label 후처리
- `preprocessing_manifest.json` 생성
- PyTorch Dataset과 sliding-window inference
- Dataset 단계의 nnU-Net v2 계열 augmentation

## 설치

GitHub에서 설치:

```bash
python -m pip install git+https://github.com/J1won7/medimg_preprocessor.git
```

로컬 소스에서 설치:

```bash
python -m pip install -e .
```

PyTorch Dataset과 augmentation까지 사용할 때:

```bash
python -m pip install "medimg-preprocessor[dataset]"
```

### Python 버전

- Python 3.8 이상을 권장합니다.
- Python 3.7도 지원하지만 `.b2nd`/`blosc2` 저장은 사용할 수 없습니다. `--storage-format npz`를 사용하십시오.
- Python 3.6 이하는 지원하지 않습니다.

Python 3.7 예시:

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir raw/imagesTr \
  --target-dir raw/labelsTr \
  --output-folder preprocessed_seg \
  --storage-format npz
```

## 입력 데이터

### 기본 디렉토리 구조

```text
dataset/
├── imagesTr/
│   ├── case_0001.nii.gz
│   └── case_0002.nii.gz
└── labelsTr/
    ├── case_0001.nii.gz
    └── case_0002.nii.gz
```

확장자를 제외한 파일명이 case identifier입니다. image와 target을 함께 사용하는
모드에서는 양쪽에 모두 존재하는 identifier만 처리하고, 한쪽에만 있는 case는
로그를 남긴 뒤 자동으로 제외합니다. 지원 확장자는 `.nii.gz`, `.nii`, `.nrrd`, `.mha`,
`.gipl`, `.tiff`, `.tif`, `.png`, `.bmp`입니다.

여러 채널을 파일로 나누었다면 `--multi-image`를 사용합니다.

```text
imagesTr/
├── case_0001_0000.nii.gz
└── case_0001_0001.nii.gz
```

### 작업 모드

| `--task-mode` | `--images-dir` | `--target-dir` | 설명 |
| --- | --- | --- | --- |
| `segmentation` | 입력 영상 | label map | semantic 또는 instance segmentation |
| `paired_generative` | source 영상 | 대응하는 target 영상 | source와 target이 1:1 대응 |
| `unpaired_generative` | domain A | domain B | 서로 대응하지 않는 두 도메인 |
| `self_supervised` | 입력 영상 | 사용하지 않음 | target 없이 학습 |

## CLI 전처리

### 최소 실행 명령

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir raw/imagesTr \
  --target-dir raw/labelsTr \
  --output-folder preprocessed_seg
```

완료 후 `--output-folder`에 전처리 case와 `preprocessing_manifest.json`이 생성됩니다.

### 필수 옵션

| 옵션 | 기능 |
| --- | --- |
| `--task-mode` | 작업 모드 선택 |
| `--images-dir` | 입력 이미지 또는 domain A 디렉토리 |
| `--target-dir` | label, paired target 또는 domain B 디렉토리 |
| `--output-folder` | 결과 디렉토리 |

`self_supervised` 모드에서는 `--target-dir`가 필요하지 않습니다.

### 공통 옵션

| 옵션 | 기본값 | 기능 |
| --- | --- | --- |
| `--spacing S1 S2 ...` | 자동 planning | target voxel spacing. spatial axis 수만큼 입력 |
| `--default-patch-size P1 P2 ...` | 자동 planning | manifest에 기록할 기본 patch 크기 |
| `--storage-format` | Python 3.8 이상: `blosc2`, Python 3.7: `npz` | 결과 저장 형식. `blosc2` 또는 `npz` |
| `--num-processes N` | CPU 절반 | planning과 preprocessing worker 수 |
| `--multi-image` | 끔 | `case_0001_0000` 형식의 multi-channel 입력 사용 |
| `--run-stage` | `train` | `train`, `predict`, `predict_and_evaluate` |
| `--val-ratio R` | `0.2` | train 단계의 validation 비율 |
| `--split-seed N` | `42` | train/validation 분할 seed |

`--spacing 1.0 1.0 1.0`처럼 입력하면 모든 축의 target spacing을 지정합니다. 원본
spacing이 `1.5 x 3.0 x 1.5`라면 축별 voxel 수는 대략 `1.5배 x 3배 x 1.5배`가
됩니다.

### Reader 옵션

| 옵션 | 사용 대상 | 가능한 값 |
| --- | --- | --- |
| `--image-reader` | `--images-dir` | `auto`, `nibabel`, `nibabel_reorient`, `simpleitk`, `simpleitk_reorient`, `tiff3d`, `natural_2d` |
| `--reference-reader` | `--target-dir` | 위와 동일 |
| `--mask-reader` | 외부 mask | 위와 동일 |

paired/unpaired 모드에서는 `--source-reader`, `--target-reader`,
`--domain-a-reader`, `--domain-b-reader`를 사용할 수 있습니다.

## 보간과 spacing

지원 보간 방식과 숫자 order는 다음과 같습니다.

| 방식 | order | 용도 |
| --- | ---: | --- |
| `nearest` | 0 | 값을 보존해야 하는 label/mask |
| `linear` | 1 | 선형 보간 |
| `quadratic` | 2 | 2차 보간 |
| `cubic` | 3 | 영상 기본값 |
| `quartic` | 4 | 4차 보간 |
| `quintic` | 5 | 5차 보간 |

자동 planning 기준 기본값은 image `cubic`, label `linear`, sampling mask `nearest`입니다.
Instance ID를 정확히 보존해야 하면 label은 `nearest`를 권장합니다.

전체 spatial axis에 같은 방식을 적용:

```bash
--image-interpolation cubic \
--label-interpolation nearest \
--mask-interpolation nearest
```

축마다 다르게 적용:

```bash
--image-interpolation-axes cubic linear cubic \
--label-interpolation-axes nearest nearest nearest \
--mask-interpolation-axes nearest nearest nearest
```

축별 옵션의 입력 순서는 transpose 이후 spatial axis 순서이며, 3D 데이터에는 세
개의 값을 입력해야 합니다. `--label-order 0`과 `--label-order 1`은 호환성을 위한
숫자 alias입니다.

- `--label-order 0`: nearest-neighbor 방식
- `--label-order 1`: 각 label ID를 독립적으로 선형 보간 후 label map으로 복원

`--label-order`와 `--label-interpolation`은 동시에 사용할 수 없습니다.

## Normalization

| 옵션 | 기능 |
| --- | --- |
| `--normalization-method auto` | dataset 정보와 planning 결과로 자동 선택 |
| `--normalization-method CTNormalization` | CT intensity의 percentile 기반 clipping과 normalization |
| `--normalization-method ZScoreNormalization` | 평균 0, 표준편차 1로 정규화 |
| `--normalization-method MinMaxClipNormalization` | 지정 범위로 clip 후 0~1 변환 |
| `--normalization-min V` | `MinMaxClipNormalization`의 하한 |
| `--normalization-max V` | `MinMaxClipNormalization`의 상한 |
| `--ct-clip-min V` | CT 자동 normalization의 고정 하한 |
| `--ct-clip-max V` | CT 자동 normalization의 고정 상한 |

CT를 `[-1000, 1000]` 범위로 정규화하는 예시:

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --images-dir raw/CBCT \
  --target-dir raw/dCT \
  --output-folder preprocessed \
  --normalization-method MinMaxClipNormalization \
  --normalization-min -1000 \
  --normalization-max 1000
```

## Mask와 patch sampling

mask는 label을 바꾸는 용도가 아니라 patch를 선택할 위치를 정하는 sampling mask입니다.
mask를 저장하지 않아도 sampling에는 사용할 수 있습니다.

### Mask 입력과 생성 옵션

| 옵션 | 기능 |
| --- | --- |
| `--images-mask-dir DIR` | image와 대응하는 외부 mask 디렉토리 |
| `--target-mask-dir DIR` | target/domain B와 대응하는 외부 mask 디렉토리 |
| `--masking-mode threshold` | threshold 기반 mask 생성 활성화 |
| `--mask-threshold V` | image와 target에 공통 threshold 적용 |
| `--images-mask-threshold V` | image에만 threshold 적용 |
| `--target-mask-threshold V` | target에만 threshold 적용 |
| `--patch-mask-min-fraction R` | `0.5` | legacy precomputed patch sampling에서 필요한 mask 비율 |
| `--patch-mask-max-starts N` | `8192` | 저장할 foreground/mask sampling 위치의 최대 개수 |
| `--save-mask` | 최종 sampling mask 저장 |
| `--no-save-mask` | sampling mask 파일 저장 안 함 |

threshold 값에 `none`을 입력하면 해당 방향의 threshold mask 생성을 끌 수 있습니다.
새로 생성되는 Dataset은 mask voxel location을 기반으로 sampling하므로
`--patch-mask-min-fraction`은 주로 기존 precomputed sampling metadata와의 호환을
위해 사용됩니다.

threshold를 사용한 unpaired 예시:

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode unpaired_generative \
  --images-dir raw/CBCT \
  --target-dir raw/dCT \
  --output-folder preprocessed \
  --masking-mode threshold \
  --images-mask-threshold 0.0 \
  --target-mask-threshold 0.0
```

segmentation에서 `--images-mask-dir`를 지정하면 kidney 같은 관심 영역을 patch
sampling 범위로 사용할 수 있습니다. target label에서 자동으로 sampling mask를
만드는 것도 가능합니다.

### Mask/label 후처리 옵션

후처리는 최종 resampling 이후 한 번 적용됩니다. 기본값은 `none`입니다.

| 옵션 | 기본값 | 기능 |
| --- | --- | --- |
| `--mask-postprocess` | `none` | binary sampling mask에 morphology 적용 |
| `--mask-closing-iters N` | `1` | mask closing 반복 횟수 |
| `--mask-keep-largest-component` | 끔 | mask의 가장 큰 연결 성분만 유지 |
| `--label-postprocess` | `none` | 각 양수 instance ID에 morphology를 독립 적용 |
| `--label-closing-iters N` | `1` | label closing 반복 횟수 |

`fill_holes`, `closing`, `fill_holes_closing`을 사용할 수 있습니다. label 후처리는
전체 label map이 아니라 각 instance ID별로 수행하며, 이미 존재하는 label voxel은
보호합니다. 새로 생성된 영역이 충돌하면 원래 instance와의 물리적 거리가 가까운
instance가 우선되고, 거리가 같으면 작은 ID가 우선됩니다.

예시:

```bash
--mask-postprocess fill_holes_closing \
--mask-closing-iters 1 \
--label-postprocess fill_holes_closing \
--label-closing-iters 1 \
--save-mask
```

`--mask-instance-postprocess`, `--mask-instance-closing-iters`,
`--label-instance-postprocess`, `--label-instance-closing-iters`는 각각 위 옵션의
호환 alias입니다.

## Segmentation과 instance segmentation

### Semantic segmentation

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir data/imagesTr \
  --target-dir data/labelsTr \
  --output-folder data/preprocessed_seg \
  --spacing 1.0 1.0 1.0 \
  --label-interpolation nearest
```

### Instance segmentation

각 객체가 서로 다른 양수 ID를 가지는 label map을 그대로 처리합니다.

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir data/imagesTr \
  --images-mask-dir data/kidney_maskTr \
  --target-dir data/instance_labelsTr \
  --output-folder data/preprocessed_instance \
  --spacing 1.0 1.0 1.0 \
  --label-interpolation nearest \
  --mask-interpolation nearest \
  --save-mask
```

## PyTorch Dataset

전처리 결과를 Dataset으로 읽으려면 dataset 의존성을 설치합니다.

```bash
python -m pip install "medimg-preprocessor[dataset]"
```

기본 사용:

```python
from torch.utils.data import DataLoader
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset(
    "preprocessed_seg",
    split="train",
)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

sample = next(iter(loader))
print(sample["image"].shape)
print(sample["target"].shape)
```

### Dataset 주요 파라미터

| 파라미터 | 기본값 | 기능 |
| --- | --- | --- |
| `folder` | 필수 | 전처리 결과 폴더 |
| `patch_size` | manifest 값 | Dataset이 반환할 최종 patch 크기 |
| `use_manifest_patch_size` | `True` | patch size를 manifest에서 자동으로 읽음 |
| `configuration` | manifest 기본값 | 여러 planning configuration 중 선택 |
| `split` | `None` | `train` 또는 `val` case만 선택 |
| `extra_folder` | `None` | conflict map/artifact prediction 같은 추가 배열 폴더 |
| `transform` | `None` | augmentation 이후 최종 sample에 적용할 사용자 transform |
| `augmentation` | `None` | augmentation 이름 또는 객체. 학습 Dataset에서만 사용 |
| `seed` | `1234` | patch sampling과 augmentation random seed |
| `patch_foreground_threshold` | `None` | threshold보다 큰 voxel을 foreground로 간주 |
| `patch_foreground_min_fraction` | `0.0` | patch 내 최소 foreground 비율 |
| `patch_foreground_source` | `image` | foreground sampling 기준: `image` 또는 `target` |
| `patch_foreground_max_tries` | `32` | foreground 조건을 만족하는 patch 탐색 횟수 |
| `random_pairing` | `True` | unpaired domain B를 무작위로 pairing |
| `view_transform` | `None` | self-supervised Dataset의 view1/view2 생성 transform |

## Dataset augmentation

augmentation은 전처리 파일을 변경하지 않고 Dataset의 `__getitem__`에서 실행됩니다.
다음과 같이 사용합니다.

```python
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset(
    "preprocessed_seg",
    split="train",
    patch_size=(32, 192, 192),
    augmentation="nnunet_v2",
)
```

`patch_size`가 있으면 다음 순서로 처리됩니다.

1. 최종 patch 위치를 sampling합니다.
2. 공간 변환에 필요한 더 큰 initial patch를 같은 중심에서 추출합니다.
3. 회전/scale 등 공간 변환을 image, target, mask에 함께 적용합니다.
4. initial patch 중앙을 최종 `patch_size`로 crop합니다.
5. intensity 변환과 mirror를 적용합니다.

initial patch는 nnU-Net의 회전/scale 범위 계산 방식을 참고해 자동으로 결정됩니다.
영상 경계를 벗어나는 initial patch 영역은 0으로 padding합니다. `patch_size`가 없으면
전체 case에 augmentation을 적용하며 별도 crop은 수행하지 않습니다.

### `NNUNetV2Augmentation` 파라미터

```python
from medimg_preprocessor import NNUNetV2Augmentation

augmentation = NNUNetV2Augmentation(
    p_rotation=0.2,
    p_scaling=0.2,
    label_interpolation="linear",
    mask_interpolation="linear",
)
```

#### 공간 변환

| 파라미터 | 기본값 | 기능 |
| --- | --- | --- |
| `p_rotation` | `0.2` | rotation을 적용할 확률 |
| `p_scaling` | `0.2` | scaling을 적용할 확률 |
| `scaling` | `(0.7, 1.4)` | scaling factor의 균등분포 범위 |
| `mirror_axes` | `None` | mirror 허용 축. `None`은 모든 축, 예: `(0, 1, 2)` |
| `dummy_2d` | `True` | anisotropic 3D patch를 in-plane 2D 변환으로 처리 |
| `anisotropy_threshold` | `3.0` | `max(shape) / shape[0]`가 이 값보다 크면 dummy-2D 적용 |

`mirror_axes`의 축 번호는 Dataset의 spatial axis 순서 기준입니다. label과 mask에는
image와 동일한 공간 변환이 적용되지만, 보간 방식은 별도로 지정됩니다.

#### Intensity 변환

| 파라미터 | 기본값 | 기능 |
| --- | --- | --- |
| `p_gaussian_noise` | `0.1` | Gaussian noise 적용 확률 |
| `noise_variance` | `(0.0, 0.1)` | noise variance 범위 |
| `p_gaussian_blur` | `0.2` | Gaussian blur 적용 확률 |
| `blur_sigma` | `(0.5, 1.0)` | blur sigma 범위 |
| `p_brightness` | `0.15` | brightness 배율 적용 확률 |
| `brightness_multiplier` | `(0.75, 1.25)` | brightness 배율 범위 |
| `p_contrast` | `0.15` | contrast 적용 확률 |
| `contrast_range` | `(0.75, 1.25)` | contrast factor 범위 |
| `p_low_resolution` | `0.25` | 저해상도 변환 후 복원할 확률 |
| `low_resolution_scale` | `(0.5, 1.0)` | 저해상도 scale 범위 |
| `p_gamma_invert` | `0.1` | gamma 변환 전 intensity 반전 확률 |
| `p_gamma` | `0.3` | gamma 변환 적용 확률 |
| `gamma_range` | `(0.7, 1.5)` | gamma 범위 |

각 `p_...` 값은 해당 변환의 적용 확률이며 `[0, 1]` 범위여야 합니다. 범위
파라미터는 `(최솟값, 최댓값)` 형식입니다.

#### Label, mask, initial patch

| 파라미터 | 기본값 | 기능 |
| --- | --- | --- |
| `label_interpolation` | `linear` | target label의 공간 보간. `linear` 또는 `nearest` |
| `mask_interpolation` | `linear` | binary mask의 공간 보간. `linear` 또는 `nearest` |
| `initial_scale_range` | `(0.85, 1.25)` | initial patch 크기 계산에 사용하는 scale 범위 |
| `paired_intensity` | `synchronized` | paired source/target intensity 변환 방식. `none`이면 paired intensity 변환을 끔 |

`linear` label 보간은 각 ID를 독립적인 binary mask로 보간하고 threshold 후 다시
label map으로 복원합니다. 객체 ID 보존이 가장 중요하면 `nearest`를 사용하십시오.
`initial_scale_range`는 실제 scaling 범위를 바꾸지 않고 initial patch 크기 계산에만
사용됩니다.

paired generative Dataset에서는 source와 target의 spatial 변환을 공유합니다.
`paired_intensity="synchronized"`이면 intensity 파라미터도 공유하고,
`paired_intensity="none"`이면 paired source/target에 intensity 변환을 적용하지 않습니다.

```python
augmentation = NNUNetV2Augmentation(
    label_interpolation="nearest",
    mask_interpolation="nearest",
    paired_intensity="none",
)

dataset = load_preprocessed_dataset(
    "preprocessed_paired",
    split="train",
    patch_size=(32, 192, 192),
    augmentation=augmentation,
)
```

사용자 정의 augmentation 객체는 `get_initial_patch_size(final_patch_size)` 메서드를
제공하면 initial patch 크기 계산에 참여할 수 있습니다. 이 메서드가 없으면 final
patch 크기로 실행됩니다. 사용자 정의 객체는 최종 crop 전까지 spatial shape을
유지해야 합니다.

validation/inference Dataset에는 augmentation을 지정하지 않는 것을 권장합니다.

## 출력 구조와 manifest

`blosc2` 저장 형식:

```text
preprocessed/
├── preprocessing_manifest.json
├── case_0001.b2nd
├── case_0001_target.b2nd
├── case_0001_mask.b2nd   # --save-mask를 사용한 경우
└── case_0001.pkl
```

`npz` 저장 형식:

```text
preprocessed/
├── preprocessing_manifest.json
├── case_0001.npz        # image, target, mask 배열을 포함할 수 있음
└── case_0001.pkl
```

manifest에는 task mode, case 목록, spacing, normalization, resampling, split,
storage format이 기록됩니다.

주요 resampling 항목:

| manifest 항목 | 의미 |
| --- | --- |
| `image_order` | 모든 축에 적용할 image order |
| `image_orders` | 축별 image order |
| `label_order` | 모든 축에 적용할 label order |
| `label_orders` | 축별 label order |
| `mask_order` | 모든 축에 적용할 mask order |
| `mask_orders` | 축별 mask order |

저장된 manifest를 확인하거나 다시 생성할 수 있습니다.

```bash
python -m medimg_preprocessor show-manifest \
  --folder preprocessed_seg

python -m medimg_preprocessor save-dataset \
  --folder preprocessed_seg \
  --task-mode segmentation
```

## 고급 설정

자동 planning 대신 설정 파일이나 nnU-Net plans를 사용할 수 있습니다.

| 옵션 | 기능 |
| --- | --- |
| `--config-json PATH` | `PreprocessingConfig` JSON 사용 |
| `--plans-file PATH` | nnU-Net plans JSON 사용 |
| `--configuration-name NAME` | plans 안의 configuration 선택 |
| `--config-a-json PATH` | unpaired domain A 설정 |
| `--config-b-json PATH` | unpaired domain B 설정 |
| `--plans-a-file PATH` | domain A plans |
| `--plans-b-file PATH` | domain B plans |
| `--configuration-a-name NAME` | domain A configuration |
| `--configuration-b-name NAME` | domain B configuration |

전체 옵션은 다음 명령으로 확인할 수 있습니다.

```bash
python -m medimg_preprocessor preprocess-dataset --help
python -m medimg_preprocessor save-dataset --help
python -m medimg_preprocessor show-manifest --help
```

## Sliding-window inference

전처리된 `.b2nd`를 추론 입력으로 사용할 필요는 없습니다. 학습 전처리 결과 폴더의
`preprocessing_manifest.json`을 읽어, 새 원본 `nii.gz`에 **동일한 spacing, transpose,
normalization, resampling 설정**을 적용하는 `ManifestInferencePatchDataset`를 사용하십시오.
manifest에 저장된 기본 patch 계획도 자동으로 사용합니다.

```python
import torch
from torch.utils.data import DataLoader
from medimg_preprocessor import ManifestInferencePatchDataset

dataset = ManifestInferencePatchDataset(
    preprocessed_folder="preprocessed_train",
    images_dir="raw/imagesTs",  # 단일 .nii.gz 파일도 가능
    overlap=0.5,
)
loader = DataLoader(dataset, batch_size=2, shuffle=False)
accumulators = dataset.build_accumulators(channels=1)

model.eval()
with torch.no_grad():
    for batch in loader:
        prediction = model(batch["image"].cuda())
        dataset.accumulate_batch(
            accumulators,
            prediction,
            batch["case_index"],
            batch["starts"],
        )

for case_index, accumulator in enumerate(accumulators):
    preprocessed_prediction = accumulator.finalize()
    dataset.save_prediction_nifti(
        preprocessed_prediction,
        case_index,
        f"predictions/{dataset.get_case(case_index).identifier}.nii.gz",
    )
```

`segmentation` manifest는 `save_prediction_nifti`에서 label용 resampling을 자동 적용하며
normalization을 역변환하지 않습니다. 모델 출력이 class logits이면 먼저 `argmax`로 label map을
만드십시오. generative task는 image resampling과 normalization 역변환을 적용합니다.

unpaired manifest는 입력 domain을 반드시 지정해야 합니다.

```python
dataset = ManifestInferencePatchDataset(
    preprocessed_folder="preprocessed_unpaired",
    images_dir="raw/domain_a_test",
    domain="a",
)
```

새 manifest는 전처리 때의 reader와 `--multi-image` 설정도 자동으로 재사용합니다. 이 정보가 없는
기존 manifest는 `auto` reader와 single-channel을 사용하므로, 필요하면 `image_reader=` 또는
`multi_image=`로 명시하십시오. manifest에 patch 계획이 없는 기존 데이터는 전처리된 영상 전체를
하나의 patch로 처리합니다. 수동 설정이 필요한 경우에는 기존
`RawInferencePatchDataset`에 `PreprocessingConfig`와 `patch_size`를 직접 전달할 수 있습니다.

## 문제 해결

### image와 target identifier가 다름

파일명에서 확장자를 제거한 case identifier가 양쪽 디렉토리에 모두 존재하는지
확인하십시오. 외부 mask는 선택 사항이며, 대응하는 case가 없는 optional mask는
사용되지 않습니다.

### `--image-interpolation-axes` 차원 오류

입력한 보간 방식 개수는 spatial dimension과 같아야 합니다. 3D 영상이면 세 개,
2D 영상이면 두 개를 입력하십시오.

### Python 3.7에서 `.b2nd` 오류

Python 3.7에서는 `blosc2` 저장을 사용할 수 없습니다. 전처리와 Dataset 모두
`--storage-format npz`로 생성된 결과를 사용하십시오.

### `_mask` 파일이 보이지 않음

mask를 patch sampling에 사용하는 것과 mask 파일을 저장하는 것은 별개입니다. 결과
파일이 필요하면 `--save-mask`를 추가하십시오.
