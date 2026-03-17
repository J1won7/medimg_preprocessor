# medimg_preprocessor

`medimg_preprocessor`는 nnU-Net의 전처리 아이디어를 독립적으로 사용할 수 있게 분리한 의료영상 전처리 라이브러리입니다.

핵심 목적은 다음입니다.

- 의료영상 헤더 정보, 특히 `spacing`을 활용한 전처리
- segmentation / paired generative / unpaired generative / self-supervised 공통 지원
- raw 파일 전처리와 PyTorch 학습용 dataset 로딩을 분리
- 잘못된 입력은 조용히 진행하지 않고 즉시 실패

## 한눈에 보기

이 라이브러리는 두 단계로 사용합니다.

1. raw 데이터를 전처리해서 `.npz`, `.pkl`, `preprocessing_manifest.json`으로 저장
2. 저장된 전처리 결과를 PyTorch `Dataset`으로 로드해서 학습

전체 흐름은 이렇습니다.

`raw files -> preprocess -> saved cases + manifest -> Dataset -> DataLoader -> training`

## 지원 범위

지원 task:

- `segmentation`
- `paired_generative`
- `unpaired_generative`
- `self_supervised`

지원 run stage:

- `train`
- `predict`
- `predict_and_evaluate`

지원 reader:

- `nii`, `nii.gz`
- `nrrd`, `mha`, `gipl` (`SimpleITK`)
- `tif`, `tiff`
- `png`, `bmp` 같은 2D 이미지

전처리 코어는 배열 기반이고, 파일 입출력은 reader가 담당합니다.

## 설치 / 의존성

기본 설치:

```bash
pip install git+https://github.com/J1won7/medimg_preprocessor.git
```

이 기본 설치에는 아래 런타임 의존성이 포함됩니다.

- `numpy`
- `scipy`
- `torch`
- `nibabel`
- `SimpleITK`
- `tifffile`
- `scikit-image`

설치 후 CLI:

```bash
python -m medimg_preprocessor --help
medimg-preprocess --help
```

## 가장 중요한 API

대부분의 사용자는 아래 네 가지만 알면 됩니다.

- `TaskAwarePreprocessor`
- `save_preprocessed_case(...)`
- `save_preprocessed_dataset(...)`
- `load_preprocessed_dataset(...)`

CLI를 쓰면 아래만 기억하면 됩니다.

- `python -m medimg_preprocessor preprocess-dataset`
- `python -m medimg_preprocessor show-manifest`

## 빠른 시작

### 1. config 만들기

직접 만들 수도 있고:

```python
from medimg_preprocessor import PreprocessingConfig, ResamplingConfig

config = PreprocessingConfig(
    spacing=(1.0, 1.0, 1.0),
    transpose_forward=(0, 1, 2),
    normalization_schemes=("zscore",),
    use_mask_for_norm=(False,),
    resampling=ResamplingConfig(image_order=3, label_order=0),
)
```

nnU-Net `plans.json`에서 가져올 수도 있습니다.

```python
from medimg_preprocessor import PreprocessingConfig

config = PreprocessingConfig.from_nnunet_plans(
    "nnUNetPlans.json",
    configuration_name="3d_fullres",
)
```

### 2. 한 케이스 전처리

배열에서 바로 처리:

```python
from medimg_preprocessor import RunStage, TaskAwarePreprocessor, TaskMode

preprocessor = TaskAwarePreprocessor(config)

case = preprocessor.run_task_case(
    image=image_array,
    image_properties={"spacing": (1.0, 1.0, 1.0)},
    task_mode=TaskMode.SEGMENTATION,
    run_stage=RunStage.TRAIN,
    reference=seg_array,
    reference_properties={"spacing": (1.0, 1.0, 1.0)},
)
```

파일에서 바로 처리:

```python
from medimg_preprocessor import NibabelIO, RunStage, TaskAwarePreprocessor, TaskMode

preprocessor = TaskAwarePreprocessor(config)

case = preprocessor.run_task_case_from_files(
    image_files=["case_0000.nii.gz"],
    image_reader=NibabelIO(),
    task_mode=TaskMode.SEGMENTATION,
    run_stage=RunStage.TRAIN,
    reference_files="case.nii.gz",
)
```

### 3. 전처리 케이스 저장

```python
from medimg_preprocessor import save_preprocessed_case

save_preprocessed_case(case, "preprocessed_seg/case_0001")
```

생성 파일:

- `preprocessed_seg/case_0001.npz`
- `preprocessed_seg/case_0001.pkl`

### 4. dataset manifest 저장

```python
from medimg_preprocessor import RunStage, TaskMode, save_preprocessed_dataset

save_preprocessed_dataset(
    folder="preprocessed_seg",
    task_mode=TaskMode.SEGMENTATION,
    run_stage=RunStage.TRAIN,
    config=config,
    default_patch_size=(96, 96, 96),
)
```

이 단계에서 `preprocessing_manifest.json`이 생성됩니다.

### 5. PyTorch dataset 로드

```python
from torch.utils.data import DataLoader
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset("preprocessed_seg")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

batch = next(iter(loader))
```

## Python 사용 가이드

### segmentation

```python
case = preprocessor.run_task_case(
    image=image_array,
    image_properties={"spacing": (1.0, 1.0, 1.0)},
    task_mode=TaskMode.SEGMENTATION,
    run_stage=RunStage.TRAIN,
    reference=seg_array,
    reference_properties={"spacing": (1.0, 1.0, 1.0)},
)
```

학습 시 batch key:

- `image`
- `target`

### paired generative

```python
case = preprocessor.run_task_case(
    image=source_array,
    image_properties={"spacing": (1.0, 1.0, 1.0)},
    task_mode=TaskMode.PAIRED_GENERATIVE,
    run_stage=RunStage.TRAIN,
    reference=target_array,
    reference_properties={"spacing": (1.0, 1.0, 1.0)},
)
```

학습 시 batch key:

- `image`
- `target`

`image`와 `target`은 같은 spatial 연산과 같은 crop을 공유합니다.

### unpaired generative

unpaired는 domain A, domain B를 각각 독립적으로 전처리합니다.

```python
case_a = preprocessor.run_task_case(
    image=image_a,
    image_properties={"spacing": (1.0, 1.0, 1.0)},
    task_mode=TaskMode.UNPAIRED_GENERATIVE,
    run_stage=RunStage.TRAIN,
)

case_b = preprocessor.run_task_case(
    image=image_b,
    image_properties={"spacing": (1.0, 1.0, 1.0)},
    task_mode=TaskMode.UNPAIRED_GENERATIVE,
    run_stage=RunStage.TRAIN,
)
```

학습 시 batch key:

- `image_a`
- `image_b`

두 도메인은 dataloader에서 랜덤하게 페어링됩니다.

### self-supervised

```python
case = preprocessor.run_task_case(
    image=image_array,
    image_properties={"spacing": (1.0, 1.0, 1.0)},
    task_mode=TaskMode.SELF_SUPERVISED,
    run_stage=RunStage.TRAIN,
)
```

기본적으로 image-only 데이터셋으로 다루고, online augmentation이나 masking은 학습 코드에서 붙이면 됩니다.

## CLI 사용 가이드

전체 help:

```powershell
python -m medimg_preprocessor --help
python -m medimg_preprocessor preprocess-dataset --help
python -m medimg_preprocessor save-dataset --help
python -m medimg_preprocessor show-manifest --help
```

### `preprocess-dataset`

raw 파일을 읽어서:

- 전처리 수행
- case 저장
- manifest 저장

까지 한 번에 처리합니다.

#### segmentation spec 예시

```json
{
  "cases": [
    {
      "identifier": "case_0001",
      "image_files": ["raw/imagesTr/case_0001_0000.nii.gz"],
      "reference_files": "raw/labelsTr/case_0001.nii.gz"
    }
  ]
}
```

실행:

```powershell
python -m medimg_preprocessor preprocess-dataset `
  --spec segmentation_spec.json `
  --output-folder preprocessed_seg `
  --task-mode segmentation `
  --run-stage train `
  --plans-file nnUNetPlans.json `
  --configuration-name 3d_fullres `
  --default-patch-size 96 96 96
```

#### unpaired spec 예시

```json
{
  "domains": {
    "a": [
      {
        "identifier": "a_0001",
        "image_files": ["raw/domain_a/a_0001.nii.gz"]
      }
    ],
    "b": [
      {
        "identifier": "b_0001",
        "image_files": ["raw/domain_b/b_0001.nii.gz"]
      }
    ]
  }
}
```

실행:

```powershell
python -m medimg_preprocessor preprocess-dataset `
  --spec unpaired_spec.json `
  --output-folder preprocessed_unpaired `
  --task-mode unpaired_generative `
  --config-a-json config_a.json `
  --config-b-json config_b.json `
  --folder-a-name domain_a `
  --folder-b-name domain_b `
  --default-patch-size 96 96 96
```

### `save-dataset`

이미 `.npz`, `.pkl` case 파일들이 만들어져 있을 때, manifest만 따로 생성합니다.

```powershell
python -m medimg_preprocessor save-dataset `
  --folder preprocessed_seg `
  --task-mode segmentation `
  --run-stage train `
  --plans-file nnUNetPlans.json `
  --configuration-name 3d_fullres `
  --default-patch-size 96 96 96
```

unpaired 예시:

```powershell
python -m medimg_preprocessor save-dataset `
  --folder preprocessed_unpaired `
  --task-mode unpaired_generative `
  --folder-a domain_a `
  --folder-b domain_b `
  --config-a-json config_a.json `
  --config-b-json config_b.json
```

### `show-manifest`

```powershell
python -m medimg_preprocessor show-manifest --folder preprocessed_seg
```

## 전처리 결과 구조

single-folder task:

```text
preprocessed_seg/
  case_0001.npz
  case_0001.pkl
  case_0002.npz
  case_0002.pkl
  preprocessing_manifest.json
```

unpaired task:

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

## 로드 후 batch key

- segmentation: `image`, `target`
- paired generative: `image`, `target`
- unpaired generative: `image_a`, `image_b`
- predict / predict_and_evaluate: `image`, 필요 시 `evaluation_reference`

## 자주 헷갈리는 점

### `save_preprocessed_dataset(...)`는 raw를 전처리하지 않습니다

Python API의 `save_preprocessed_dataset(...)`는 manifest 저장 함수입니다.

raw를 읽어서 전처리까지 한 번에 하려면:

- Python에서는 `TaskAwarePreprocessor` + `save_preprocessed_case(...)`
- CLI에서는 `preprocess-dataset`

을 사용해야 합니다.

### unpaired는 1:1 매칭이 아닙니다

domain A와 B는 독립적으로 저장되고, 학습 시 랜덤하게 페어링됩니다.

### fail-fast 정책입니다

입력이 잘못되면 warning 후 즉시 예외를 냅니다. 조용히 진행하지 않습니다.

예:

- image/target shape mismatch
- paired source/target spacing mismatch
- CT normalization 통계 누락
- 잘못된 patch 차원
- manifest 누락

## 권장 사용 패턴

일반적으로는 아래 순서를 권장합니다.

1. `plans.json` 또는 직접 config로 전처리 설정 고정
2. raw dataset 전체를 offline preprocessing
3. `load_preprocessed_dataset(...)`로 학습 dataset 생성
4. PyTorch `DataLoader`에 연결
5. 모델 학습
