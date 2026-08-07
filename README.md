# medimg_preprocessor

CT, MRI, CBCT와 같은 의료 영상을 위한 범용 전처리 도구입니다. nnU-Net의
planning 아이디어를 참고하지만, 특정 학습 프레임워크에 종속되지 않도록
segmentation과 생성 모델 전처리를 함께 지원합니다.

주요 기능:

- 데이터셋 자동 스캔과 전처리 설정 계획
- segmentation, instance segmentation, paired/unpaired generative 전처리
- 외부 mask 또는 threshold 기반 patch sampling mask 생성
- 원하는 voxel spacing으로 영상과 label/mask resampling
- 전처리 결과 manifest 생성
- PyTorch Dataset 및 sliding-window inference 지원

## 설치

```bash
python -m pip install git+https://github.com/J1won7/medimg_preprocessor.git
```

로컬 소스에서 개발하려면:

```bash
python -m pip install -e .
```

### Python 버전

- Python 3.8 이상을 권장합니다.
- Python 3.7도 사용할 수 있지만 `blosc2` 저장 형식은 사용할 수 없습니다. 이 경우 `npz`를 사용해야 합니다.
- Python 3.7에서 전처리할 때는 `--storage-format npz`를 명시하는 것이 안전합니다.
- Python 3.6 이하는 지원하지 않습니다.

Python 3.7에서 생성된 `npz` 결과는 Python 3.8 이상에서도 읽을 수 있습니다. 반대로
Python 3.7 환경에서는 Python 3.8 이상에서 생성한 `blosc2`/`.b2nd` 결과를 읽을 수 없습니다.

## 가장 간단한 사용법

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir raw/imagesTr \
  --target-dir raw/labelsTr \
  --output-folder preprocessed_seg
```

실행이 끝나면 `preprocessing_manifest.json`과 전처리된 case 파일이
`--output-folder` 아래에 저장됩니다.

## Task 모드

| 모드 | `--images-dir` | `--target-dir` | 용도 |
| --- | --- | --- | --- |
| `segmentation` | 입력 영상 | segmentation label | semantic 또는 instance segmentation |
| `paired_generative` | source 영상 | paired target 영상 | source와 target이 일대일 대응하는 생성 모델 |
| `unpaired_generative` | domain A | domain B | 서로 대응하지 않는 두 도메인 |
| `self_supervised` | 입력 영상 | 사용하지 않음 | self-supervised 학습 |

`--task-mode`에 따라 필요한 디렉토리만 지정하면 됩니다.

## 입력 파일 규칙

기본적으로 이미지와 target/mask는 파일명에서 확장자를 제거한 case identifier가
같아야 합니다. 예를 들어 다음 두 파일은 같은 case로 인식됩니다.

```text
imagesTr/case_0001.nii.gz
labelsTr/case_0001.nii.gz
```

여러 채널을 파일로 나누어 저장한 경우에는 `--multi-image`를 사용합니다.

```text
case_0001_0000.nii.gz
case_0001_0001.nii.gz
```

## Segmentation과 Instance Segmentation

`segmentation` 모드는 semantic segmentation뿐 아니라 각 객체에 서로 다른 label
번호가 들어 있는 instance label도 처리할 수 있습니다. target 파일의 각 voxel 값은
그대로 label map으로 취급됩니다.

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

instance ID를 보존해야 한다면 label은 nearest-neighbor를 권장합니다.

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

`--label-interpolation linear`을 사용하면 각 label ID를 binary mask로 분리한 뒤
각각 보간하고 다시 label map으로 합칩니다. 경계가 부드러워질 수 있지만 얇은 구조나
서로 가까운 instance가 변할 수 있으므로, 객체 ID 보존이 중요하면 `nearest`가 더 안전합니다.

## Mask와 Patch Sampling

mask는 target label을 바꾸기 위한 mask가 아니라, patch를 어디에서 추출할지 결정하는
sampling mask입니다. mask가 있으면 patch 중심 voxel을 mask 내부에서 선택합니다.

mask 생성 규칙:

- 외부 mask와 threshold mask가 함께 있으면 두 mask를 합쳐서 사용합니다.
- 외부/threshold mask가 하나도 없고 segmentation target이 있으면 target label에서 mask를 만듭니다.
- 그 외 모드에서 mask가 없으면 전체 영상 범위에서 patch를 sampling합니다.

### 외부 mask

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --images-dir raw/source \
  --target-dir raw/target \
  --images-mask-dir raw/source_masks \
  --target-mask-dir raw/target_masks \
  --output-folder preprocessed_paired
```

- `--images-mask-dir`: `--images-dir`와 대응하는 mask 디렉토리
- `--target-mask-dir`: `--target-dir`와 대응하는 mask 디렉토리

### Threshold mask

외부 mask가 없을 때 threshold 기반 mask를 만들 수 있습니다.

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

관련 옵션:

- `--mask-threshold`: image와 target 양쪽에 공통 적용
- `--images-mask-threshold`: image 쪽에만 적용
- `--target-mask-threshold`: target 쪽에만 적용
- `none`: 해당 쪽 threshold mask를 명시적으로 끔

### 최종 mask/label 후처리

mask와 label은 모두 최종 spacing으로 리샘플링한 뒤 후처리를 한 번만 적용합니다.
기본값은 `none`이며, 명시적으로 켠 경우에만 원본 mask/label이 변경됩니다.

후처리 방식은 `none`, `fill_holes`, `closing`, `fill_holes_closing` 중에서 선택합니다.

- `--mask-postprocess`: 최종 binary sampling mask에 적용
- `--mask-closing-iters N`: mask closing 반복 횟수
- `--mask-keep-largest-component`: 최종 mask에서 가장 큰 연결 성분만 유지
- `--label-postprocess`: 각 양수 label ID에 독립적으로 적용
- `--label-closing-iters N`: label closing 반복 횟수

예를 들어 mask와 label에 각각 hole filling과 closing을 적용하려면 다음과 같이
실행합니다.

```bash
--mask-postprocess fill_holes_closing \
--mask-closing-iters 1 \
--label-postprocess fill_holes_closing \
--label-closing-iters 1
```

label은 전체 map을 하나의 binary mask로 처리하지 않고, 각 ID를 별도 binary mask로
변환한 뒤 다시 합칩니다. 원래 label voxel은 보호하며, 새로 확장된 영역이 여러
instance에 동시에 해당하면 최종 spacing을 반영한 물리적 거리가 가까운 ID를 선택하고,
거리가 같으면 작은 ID를 선택합니다. `fill_holes`는 의도적인 내부 cavity도 채울 수
있고, `closing`은 객체를 확장할 수 있으므로 instance label에만 사용하십시오.

### 저장되는 `_mask` 파일

`_mask`는 patch sampling용 mask입니다. normalization에서 사용하는
`use_mask_for_norm`과는 다른 기능입니다.

- `--save-mask`: mask를 결과에 저장. blosc2에서는 `_mask.b2nd`, npz에서는 `.npz` 내부 배열로 저장
- `--no-save-mask`: mask 파일을 저장하지 않음

파일을 저장하지 않아도 patch sampling 자체는 mask를 사용합니다. segmentation 학습
단계에서는 기본적으로 `_mask` 파일을 저장하지 않지만, 필요하면 `--save-mask`를 사용하십시오.

## Cropping 동작

이 도구는 더 이상 nnU-Net식 `crop_to_nonzero`를 적용하지 않습니다.

- 원래 image/target의 spatial extent를 유지합니다.
- image-derived crop 바깥 label을 `-1`로 바꾸지 않습니다.
- 전처리 후 label 의미가 원본 데이터와 동일하게 유지됩니다.

따라서 patch를 mask 내부에서 sampling할 수는 있지만, 전처리 단계에서 영상 자체를
자동으로 잘라내지는 않습니다.

## Voxel Spacing과 보간

`--spacing`에는 원하는 target voxel spacing을 공간축 순서대로 입력합니다.

```bash
--spacing 1.0 1.0 1.0
```

예를 들어 원본 spacing이 `1.5 x 3.0 x 1.5`이고 target이 `1.0 x 1.0 x 1.0`이면,
각 축의 voxel 수가 각각 약 1.5배, 3배, 1.5배로 변경됩니다. 영상, label, mask 모두
동일한 공간축 shape 변환을 거치며, 각 축의 보간 방식은 전체 또는 축별 옵션으로
명시합니다.

### 보간 방식 지정

지원 방식은 `nearest`, `linear`, `quadratic`, `cubic`, `quartic`, `quintic`입니다.

기본값:

- image: cubic
- 자동 planning된 segmentation label: linear
- sampling mask: nearest

전체 공간축에 같은 방식을 적용하려면:

```bash
--image-interpolation cubic
--label-interpolation nearest
--mask-interpolation nearest
```

축별로 다르게 지정하려면 spatial axis 개수만큼 입력합니다. 입력 순서는 transpose
이후의 spatial axis 순서입니다.

```bash
--image-interpolation-axes cubic linear cubic
--label-interpolation-axes nearest nearest nearest
--mask-interpolation-axes nearest linear nearest
```

기존 숫자 옵션인 `--label-order 0`, `--label-order 1`도 호환성을 위해 지원하지만,
새 명령에서는 의미가 분명한 `--label-interpolation` 사용을 권장합니다.

## Normalization

자동 planning을 그대로 사용하거나 normalization을 직접 지정할 수 있습니다.

```bash
--normalization-method auto
--normalization-method CTNormalization
--normalization-method ZScoreNormalization
--normalization-method MinMaxClipNormalization
```

CT에서 고정 범위로 clip하려면:

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

## 자주 사용하는 옵션

- `--num-processes N`: planning과 전처리에 사용할 worker 수
- `--run-stage train|predict|predict_and_evaluate`: 실행 단계
- `--default-patch-size 96 96 96`: planner가 정한 patch size 대신 사용할 크기
- `--multi-image`: 한 case가 여러 파일로 구성된 multi-channel 입력일 때 사용
- `--storage-format blosc2|npz`: 결과 저장 형식
- `--config-json PATH`: 직접 작성한 preprocessing config 사용
- `--plans-file PATH --configuration-name NAME`: nnU-Net 스타일 plans 사용

## 출력 구조와 Manifest

blosc2 저장 형식:

```text
preprocessed/
├── preprocessing_manifest.json
├── case_0001.b2nd
├── case_0001_target.b2nd
├── case_0001_mask.b2nd     # --save-mask를 사용한 경우
└── case_0001.pkl
```

`npz` 저장 형식:

```text
preprocessed/
├── preprocessing_manifest.json
├── case_0001.npz           # image, target, mask 배열을 포함할 수 있음
└── case_0001.pkl
```

`preprocessing_manifest.json`에는 task mode, case 목록, spacing, normalization,
resampling, split, 저장 형식이 기록됩니다.

resampling 설정은 다음처럼 저장됩니다.

- `image_order`: 모든 축에 적용할 image 보간 order
- `image_orders`: 축별 image 보간 order
- `label_order`: 모든 축에 적용할 label 보간 order
- `label_orders`: 축별 label 보간 order
- `mask_order`: 모든 축에 적용할 sampling mask 보간 order
- `mask_orders`: 축별 sampling mask 보간 order

보간 order는 `0=nearest`, `1=linear`, `2=quadratic`, `3=cubic`의 의미입니다.
축별 설정을 사용하지 않으면 지정한 하나의 order가 모든 공간축에 적용됩니다.

manifest만 다시 생성해야 하는 경우:

```bash
python -m medimg_preprocessor save-dataset \
  --folder preprocessed_seg \
  --task-mode segmentation
```

manifest 내용을 확인하려면:

```bash
python -m medimg_preprocessor show-manifest \
  --folder preprocessed_seg
```

## PyTorch Dataset 사용

```bash
python -m pip install "medimg-preprocessor[dataset]"
```

```python
from torch.utils.data import DataLoader
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset("preprocessed_seg", split="train")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

sample = next(iter(loader))
print(sample["image"].shape)
print(sample["target"].shape)
```

다음 dataset class도 직접 사용할 수 있습니다.

- `SegmentationDataset`
- `PairedGenerativeDataset`
- `UnpairedGenerativeDataset`
- `SelfSupervisedDataset`

## Inference

원본 영상에서 runtime preprocessing과 sliding-window inference를 수행하려면
`RawInferencePatchDataset`를 사용합니다.

```python
from torch.utils.data import DataLoader
from medimg_preprocessor import PreprocessingConfig, RawInferencePatchDataset

config = PreprocessingConfig(
    spacing=(1.0, 1.0, 1.0),
    transpose_forward=(0, 1, 2),
    normalization_schemes=("ZScoreNormalization",),
    use_mask_for_norm=(False,),
)

dataset = RawInferencePatchDataset(
    images_dir="raw/images",
    config=config,
    patch_size=(32, 192, 192),
    overlap=0.5,
)
loader = DataLoader(dataset, batch_size=2, shuffle=False)
```

patch prediction을 합친 뒤 `save_prediction_nifti`를 사용하면 원본 NIfTI의 affine과
header를 유지한 결과를 저장할 수 있습니다.

## 문제 해결

### `images and labels must contain the same case identifiers`

image와 target 파일의 case identifier가 일치하지 않는 경우입니다. 파일명과 확장자를
확인하고, 일부 case가 한쪽에만 있다면 해당 파일을 정리한 뒤 다시 실행하십시오.

### `--image-interpolation-axes` 차원 오류

입력한 보간 방식 개수가 영상의 spatial dimension과 같아야 합니다. 3D 영상이면 세 개,
2D 영상이면 두 개를 입력해야 합니다.

### Python 3.7에서 `blosc2` 오류

Python 3.7에서는 `blosc2`/`.b2nd`를 사용할 수 없습니다. 다음처럼 `npz`를 지정하십시오.

```bash
python -m medimg_preprocessor preprocess-dataset \
  ... \
  --storage-format npz
```

### `_mask`가 저장되지 않음

mask를 patch sampling에 사용하는 것과 mask 파일을 디스크에 저장하는 것은 별개입니다.
파일도 필요하면 `--save-mask`를 추가하십시오.
