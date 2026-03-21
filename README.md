# medimg_preprocessor

`medimg_preprocessor`는 nnU-Net의 전처리 철학을 최대한 유지하면서, 다른 프로젝트에서도 더 쉽게 재사용할 수 있도록 분리한 의료영상 전처리 라이브러리입니다.

핵심 목표는 아래와 같습니다.

- nnU-Net과 비슷한 전처리 흐름 유지
- CLI 사용 단순화
- PyTorch 학습 코드에 바로 연결 가능한 저장 포맷 제공
- segmentation, paired generative, unpaired generative, self-supervised를 한 인터페이스로 지원

## 무엇을 자동으로 해주나

`preprocess-dataset`를 실행하면, 별도 config를 주지 않아도 아래를 자동으로 수행합니다.

- 영상 헤더에서 voxel spacing 읽기
- nonzero crop
- dataset fingerprint 추출
- nnU-Net 스타일 target spacing 결정
- transpose 순서 결정
- channel name 기반 normalization scheme 결정
- `2d / 3d` 기본 configuration 생성
- configuration별 `patch_size` 계산
- configuration별 `recommended_batch_size` 계산
- train 단계에서는 자동 `train / val` split 생성
- 전처리 결과 저장
- `preprocessing_manifest.json` 생성

즉 사용자 입장에서는 아래 정도만 입력하면 됩니다.

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir imagesTr \
  --labels-dir labelsTr \
  --output-folder preprocessed
```

## 지원 범위

현재 기준으로 전처리와 데이터 로딩에 집중합니다.

포함되는 것:

- `nii`, `nii.gz`, `nrrd`, `mha`, `gipl`, `tif`, `tiff`, `png`, `bmp` reader
- header 기반 spacing 처리
- nonzero crop
- anisotropy-aware resampling
- normalization planning
- `2d / 3d` patch planning
- `blosc2` 기반 저장
- PyTorch dataset 로드

포함하지 않는 것:

- nnU-Net 전체 trainer 복제
- `3d_lowres`, `cascade`
- nnU-Net의 전체 architecture planner

즉 이 라이브러리는 `nnU-Net preprocessing/planning 중심`이지, `nnU-Net 전체 프레임워크 복제`는 아닙니다.

## 설치

```bash
pip install git+https://github.com/J1won7/medimg_preprocessor.git
```

기본 의존성은 패키지에 포함되어 있습니다.

- `numpy`
- `scipy`
- `torch`
- `blosc2`
- `nibabel`
- `SimpleITK`
- `tifffile`
- `scikit-image`

## 전체 사용 흐름

```text
raw 디렉토리
-> preprocess-dataset
-> 전처리 파일(.b2nd/.pkl) + preprocessing_manifest.json
-> load_preprocessed_dataset(...)
-> PyTorch DataLoader
-> 학습
```

## 가장 많이 쓰는 명령

### 1. Segmentation

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode segmentation \
  --images-dir imagesTr \
  --labels-dir labelsTr \
  --output-folder preprocessed \
  --num-processes 8
```

### 2. Paired Generative

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --source-dir source \
  --target-dir target \
  --output-folder preprocessed_paired
```

### 3. Unpaired Generative

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode unpaired_generative \
  --domain-a-dir domain_a \
  --domain-b-dir domain_b \
  --output-folder preprocessed_unpaired
```

### 4. Self-Supervised

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode self_supervised \
  --images-dir images \
  --output-folder preprocessed_ssl
```

## 정규화 방식 직접 지정

`preprocess-dataset` 실행 시 자동 planning 대신 아래 3가지 정규화 방식 중 하나를 강제로 적용할 수 있습니다.

- `--normalization-method CTNormalization`
  - nnU-Net의 CT 정규화와 같은 방식입니다.
  - 각 채널에 대해 dataset-level 통계를 사용합니다.
  - 먼저 clip을 적용한 뒤 `mean/std`로 정규화합니다.
  - 현재 구현에서는 `clip_min/clip_max`가 있으면 그 값을 쓰고, 없으면 `percentile_00_5/percentile_99_5`를 사용합니다.

- `--normalization-method ZScoreNormalization`
  - 각 이미지 채널별로 z-score 정규화를 적용합니다.
  - 즉 `mean`을 빼고 `std`로 나눕니다.
  - mask 기반 정규화가 설정되어 있으면 nonzero 영역 또는 reference mask 기준으로 계산합니다.

- `--normalization-method MinMaxClipNormalization`
  - 사용자가 지정한 `min/max` 범위로 먼저 clip합니다.
  - 그 다음 `[0, 1]` 범위로 선형 스케일링합니다.
  - 이 모드를 사용할 때는 반드시 `--normalization-min`, `--normalization-max`를 함께 지정해야 합니다.

예시:

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --source-dir source \
  --target-dir target \
  --output-folder preprocessed_paired \
  --normalization-method ZScoreNormalization
```

```bash
python -m medimg_preprocessor preprocess-dataset \
  --task-mode paired_generative \
  --source-dir source \
  --target-dir target \
  --output-folder preprocessed_paired \
  --normalization-method MinMaxClipNormalization \
  --normalization-min -1000 \
  --normalization-max 2000
```

## 디렉토리 규칙

### Segmentation

입력:

- `--images-dir`
- `--labels-dir`

자동 매칭 기준:

- 같은 case identifier

예시:

- `imagesTr/case_0001_0000.nii.gz`
- `labelsTr/case_0001.nii.gz`

### Paired Generative

입력:

- `--source-dir`
- `--target-dir`

자동 매칭 기준:

- 같은 case identifier

예시:

- `source/case_0001.nii.gz`
- `target/case_0001.nii.gz`

또는

- `source/case_0001_0000.nii.gz`
- `target/case_0001_0000.nii.gz`

### Unpaired Generative

입력:

- `--domain-a-dir`
- `--domain-b-dir`

특징:

- A와 B는 preprocessing 시점에는 1:1 매칭하지 않음
- 학습 시 dataset이 랜덤 pairing

### Self-Supervised

입력:

- `--images-dir`

특징:

- image만 전처리
- target은 학습 루프에서 online 생성

## Multi-image 규칙

한 케이스가 여러 파일로 구성되어 있으면 `--multi-image`를 사용합니다.

예시:

- `case_0001_0000.nii.gz`
- `case_0001_0001.nii.gz`
- `case_0001_0002.nii.gz`

규칙:

- postfix는 4자리 숫자여야 함
- `0000`부터 시작해야 함
- 중간 번호가 비면 안 됨

적용 대상:

- `--images-dir`
- `--source-dir`
- `--target-dir`
- `--domain-a-dir`
- `--domain-b-dir`

주의:

- segmentation label은 여전히 case당 1파일입니다

## 자동 planning 결과는 어디에 저장되나

전처리 후 생성되는 `preprocessing_manifest.json` 안에 들어갑니다.

이 파일에는 보통 아래 정보가 포함됩니다.

- `task_mode`
- `run_stage`
- `preprocessing_config`
- `default_configuration`
- `configurations["2d"]`
- `configurations["3d"]`
- `default_patch_size`
- `splits["train"]`, `splits["val"]`
- `storage_format`

즉 `nnU-Net의 plans.json과 비슷한 역할을 현재 라이브러리에서는 manifest가 담당`한다고 보면 됩니다.

## 출력 구조

### Single-folder task

```text
preprocessed/
  case_0001.b2nd
  case_0001_target.b2nd        # 필요한 경우만
  case_0001_evalref.b2nd       # 필요한 경우만
  case_0001.pkl
  case_0002.b2nd
  case_0002.pkl
  preprocessing_manifest.json
```

### Unpaired task

```text
preprocessed_unpaired/
  domain_a/
    a_0001.b2nd
    a_0001.pkl
  domain_b/
    b_0001.b2nd
    b_0001.pkl
  preprocessing_manifest.json
```

## PyTorch에서 로드하는 방법

### 기본 로드

```python
from torch.utils.data import DataLoader
from medimg_preprocessor import load_preprocessed_dataset

dataset = load_preprocessed_dataset("preprocessed")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

batch = next(iter(loader))
```

### 2D configuration으로 로드

자동 planning 결과에 `2d`가 있으면 아래처럼 바로 선택할 수 있습니다.

```python
from medimg_preprocessor import load_preprocessed_dataset

dataset_2d = load_preprocessed_dataset("preprocessed", configuration="2d")
dataset_3d = load_preprocessed_dataset("preprocessed", configuration="3d")
```

동작:

- `configuration="3d"`: 3D patch 사용
- `configuration="2d"`: 3D volume이면 slice-wise 2D patch 사용

### patch_size 직접 override

```python
dataset = load_preprocessed_dataset(
    "preprocessed",
    patch_size=(96, 96, 96),
)
```

우선순위:

1. `load_preprocessed_dataset(..., patch_size=...)`
2. `configuration="2d" | "3d"`에 저장된 patch size
3. manifest의 `default_patch_size`

### train / val split으로 로드

train 단계로 전처리했다면 manifest에 split이 자동 저장됩니다.

```python
from medimg_preprocessor import load_preprocessed_dataset

train_dataset = load_preprocessed_dataset("preprocessed", split="train")
val_dataset = load_preprocessed_dataset("preprocessed", split="val")
```

unpaired도 동일하게 사용할 수 있습니다.

```python
train_dataset = load_preprocessed_dataset("preprocessed_unpaired", split="train")
val_dataset = load_preprocessed_dataset("preprocessed_unpaired", split="val")
```

## task별 반환 키

### Segmentation

- `image`
- `target`
- `identifier`
- `properties`

### Paired Generative

- `image`
- `target`
- `identifier`
- `properties`

### Unpaired Generative

- `image_a`
- `image_b`
- `identifier_a`
- `identifier_b`
- `properties_a`
- `properties_b`

### Predict / Predict-and-Evaluate

- `image`
- optional `evaluation_reference`

## config를 직접 주고 싶을 때

자동 planning이 기본입니다. 필요하면 아래 둘 중 하나로 override할 수 있습니다.

### 1. 직접 config JSON 사용

```bash
--config-json config.json
```

### 2. nnU-Net plans 사용

```bash
--plans-file nnUNetPlans.json --configuration-name 3d_fullres
```

unpaired에서는 다음도 가능합니다.

- `--config-a-json`, `--config-b-json`
- `--plans-a-file`, `--plans-b-file`

## 저장 포맷

기본 저장 포맷은 `blosc2`입니다.

기본 출력:

- `case_0001.b2nd`
- optional `case_0001_target.b2nd`
- optional `case_0001_evalref.b2nd`
- `case_0001.pkl`

이전 NumPy archive 포맷을 강제로 쓰고 싶으면:

```bash
--storage-format npz
```

## 병렬 처리

planning과 preprocessing 모두 multi-process로 돌릴 수 있습니다.

```bash
--num-processes 8
```

지정하지 않으면 기본적으로 CPU 코어의 절반을 사용합니다.

## manifest만 따로 만들고 싶을 때

이미 전처리 case 파일이 있고 `preprocessing_manifest.json`만 다시 만들고 싶으면:

```bash
python -m medimg_preprocessor save-dataset --help
```

## 저장된 설정 확인

```bash
python -m medimg_preprocessor show-manifest --folder preprocessed
```

## CLI 도움말

```bash
python -m medimg_preprocessor --help
python -m medimg_preprocessor preprocess-dataset --help
python -m medimg_preprocessor save-dataset --help
python -m medimg_preprocessor show-manifest --help
```

설치 후에는 아래도 가능합니다.

```bash
medimg-preprocess --help
```

## Fail-fast 정책

이 라이브러리는 조용히 진행하지 않고, 잘못된 입력이면 바로 실패하도록 설계되어 있습니다.

예:

- image/label identifier 불일치
- source/target identifier 불일치
- 잘못된 multi-image postfix
- shape/spacings 불일치
- invalid config
- 필요한 target 누락

즉 잘못된 사용을 억지로 허용하기보다, 빨리 실패해서 원인을 바로 알 수 있게 하는 쪽입니다.

## 현재 위치

이 라이브러리는 아래에 가깝습니다.

- nnU-Net preprocessing/planning과 최대한 비슷한 동작
- 더 단순한 CLI
- custom PyTorch 코드와 연결하기 쉬운 저장/로딩 구조
- generative / unpaired workflow까지 한 인터페이스로 확장

완전한 nnU-Net 복제는 아니지만, `전처리와 planning을 다른 프로젝트에서 편하게 재사용`하는 목적에는 맞게 정리되어 있습니다.
