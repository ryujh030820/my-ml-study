# NeRF (Neural Radiance Fields) 구현

PyTorch를 사용한 간단한 NeRF 구현입니다. Mac의 Metal Performance Shaders (MPS)를 활용하여 GPU 가속 학습을 지원합니다.

## 📋 개요

NeRF는 2D 이미지들로부터 3D 장면을 재구성하는 신경망 기반 방법입니다. 이 구현은 다음을 포함합니다:

-   **모델 구조**: Positional encoding과 view-dependent 색상을 사용하는 MLP
-   **볼륨 렌더링**: Ray marching을 통한 이미지 합성
-   **MPS 가속**: Apple Silicon에서의 빠른 학습
-   **합성 데이터**: 실제 데이터셋 없이도 테스트 가능

## 🔧 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 필요한 패키지

-   PyTorch 2.0+ (MPS 지원)
-   NumPy
-   Pillow
-   Matplotlib
-   tqdm

## 🚀 사용법

### 1. 빠른 데모 (5분)

5분 안에 NeRF를 체험해보세요! (축소된 모델, 빠른 학습)

```bash
python quick_demo.py
```

이 데모는 간단한 합성 데이터로 20 에폭만 학습하여 결과를 보여줍니다.

### 2. 실제 데이터셋으로 학습 (권장)

#### 데이터셋 다운로드

NeRF synthetic dataset을 다운로드하세요:

```bash
# 방법 1: Google Drive에서 직접 다운로드
# https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1

# 방법 2: gdown 사용
pip install gdown
gdown 1lDkKNdpoZEja3zWDXBdZjfdaB0HxH_EX
unzip nerf_synthetic.zip
```

데이터셋을 `./nerf_synthetic/` 폴더에 배치하세요.

#### 학습 실행

원하는 scene을 선택해서 학습:

```bash
# Lego scene 학습 (기본값)
python train_real.py lego

# 다른 scene 학습
python train_real.py chair
python train_real.py drums
python train_real.py ficus
python train_real.py hotdog
python train_real.py materials
python train_real.py mic
python train_real.py ship
```

#### 테스트/렌더링

학습된 모델로 새로운 뷰 렌더링:

```bash
# Lego scene 테스트
python test_real.py lego

# 다른 데이터셋으로 테스트 (train/val/test)
python test_real.py lego test
python test_real.py chair val
```

### 3. 체크포인트에서 이어서 학습

학습 중단 후 재개하거나 추가 학습:

```bash
# 가장 최근 체크포인트에서 자동 재개
python resume_train.py lego

# 특정 체크포인트 선택
python resume_train.py lego nerf_epoch_20.pth

# 또는 직접 실행
python train_real.py lego nerf_final.pth

# 사용 가능한 체크포인트 확인
python resume_train.py lego list
```

자세한 내용은 `RESUME_GUIDE.md`를 참고하세요.

### 4. 간단한 합성 데이터로 학습

실제 데이터셋 없이 테스트:

```bash
# 샘플 데이터 생성
python create_sample_data.py

# 학습
python train.py
```

### 학습 파라미터 조정

`train.py`의 `main()` 함수에서 다음 파라미터를 조정할 수 있습니다:

-   `n_epochs`: 학습 에폭 수 (기본값: 50)
-   `batch_size`: 배치당 ray 수 (기본값: 1024)
-   `lr`: learning rate (기본값: 5e-4)
-   `N_samples`: ray당 샘플링 포인트 수 (기본값: 64)

### 테스트/렌더링

학습된 모델로 새로운 뷰를 렌더링:

```bash
python test.py
```

테스트 스크립트는 다음을 생성합니다:

-   단일 뷰 렌더링 (RGB + 깊이 맵)
-   360도 회전 비디오 프레임들
-   결과 시각화

## 📁 프로젝트 구조

```
NeRF/
├── model.py              # NeRF 모델 및 렌더링 함수
├── data.py               # 데이터 로더 및 합성 데이터 생성
├── config.py             # 데이터셋별 설정 (NEW)
├── train.py              # 학습 스크립트 (간단한 데이터)
├── train_real.py         # 실제 데이터셋 학습 스크립트 (NEW)
├── test.py               # 테스트/렌더링 스크립트
├── test_real.py          # 실제 데이터셋 테스트 스크립트 (NEW)
├── resume_train.py       # 체크포인트 재개 스크립트 (NEW)
├── quick_demo.py         # 빠른 데모 스크립트 (5분 체험)
├── create_sample_data.py # 샘플 데이터 생성 스크립트 (NEW)
├── download_data.py      # 데이터셋 다운로드 가이드
├── run.sh                # 전체 파이프라인 실행 스크립트
├── run_real.sh           # 실제 데이터셋 파이프라인 (NEW)
├── requirements.txt      # 의존성 목록
├── README.md             # 이 파일
├── USAGE.md              # 사용 가이드 (NEW)
├── RESUME_GUIDE.md       # 체크포인트 재개 가이드 (NEW)
├── nerf_synthetic/       # 실제 NeRF dataset (다운로드 필요)
├── checkpoints/          # 학습된 모델 저장 (자동 생성)
└── renders/              # 렌더링 결과 저장 (자동 생성)
```

## 🎯 주요 기능

### 1. NeRF 모델 (`model.py`)

-   **Positional Encoding**: 고주파 디테일 학습을 위한 주파수 인코딩
-   **View-dependent 색상**: 방향에 따른 반사 효과 표현
-   **Skip Connection**: 깊은 네트워크 학습 안정화

### 2. 볼륨 렌더링

-   **Stratified Sampling**: 균등 샘플링과 랜덤 샘플링 결합
-   **Alpha Compositing**: 투명도 기반 색상 합성
-   **깊이 맵 생성**: 3D 구조 시각화

### 3. 디바이스 지원

-   **MPS (Metal)**: Apple Silicon (M1/M2/M3) GPU 가속
-   **CUDA**: NVIDIA GPU 지원
-   **CPU**: fallback 옵션

## 📊 학습 과정

학습 중에는 다음이 저장됩니다:

1. **체크포인트**: `checkpoints/nerf_epoch_X.pth` (주기적으로)
2. **최종 모델**: `checkpoints/nerf_final.pth`
3. **Loss 그래프**: `checkpoints/training_loss.png`

## 🎨 렌더링 결과

테스트 후 `renders/` 디렉토리에 다음이 저장됩니다:

-   `test_render.png`: RGB와 깊이 맵 비교
-   `test_rgb.png`: 렌더링된 RGB 이미지
-   `frame_XXX.png`: 360도 회전 비디오 프레임들
-   `results.png`: 여러 프레임 시각화

## ⚙️ 성능 최적화

### Mac에서의 MPS 사용

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
```

### 배치 크기 조정

-   GPU 메모리에 따라 `batch_size` 조정
-   M1/M2: 1024-2048 rays
-   M3: 2048-4096 rays

### 샘플링 포인트 수

-   빠른 학습: `N_samples=32`
-   균형: `N_samples=64` (기본값)
-   고품질: `N_samples=128`

## 📚 참고 자료

-   **원본 논문**: [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
-   **공식 구현**: [bmild/nerf](https://github.com/bmild/nerf)
-   **PyTorch 구현**: [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)

## 🔍 알려진 제한사항

1. 간단한 구현이므로 원본 논문의 모든 최적화를 포함하지 않습니다
2. Hierarchical sampling (coarse-fine) 미구현
3. 실제 데이터셋 사용 시 전처리 필요

## 🛠️ 트러블슈팅

### MPS 오류 발생 시

```python
# train.py 또는 test.py에서 device를 CPU로 변경
device = torch.device("cpu")
```

### 메모리 부족 시

-   `batch_size` 감소 (예: 512)
-   `N_samples` 감소 (예: 32)
-   이미지 해상도 감소 (H, W를 100x100으로)

## 📝 라이선스

이 코드는 교육 목적으로 작성되었습니다.
