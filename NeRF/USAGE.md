# NeRF 사용 가이드

## 빠른 시작

### 1️⃣ 환경 설정

```bash
cd /Users/junghwanryu/my-ml-study/NeRF
pip install -r requirements.txt
```

### 2️⃣ 빠른 데모 (5분)

```bash
python quick_demo.py
```

결과: `demo_result.png`

---

## 실제 데이터셋 사용하기

### 데이터셋 다운로드

이미 `nerf_synthetic/` 폴더가 있으므로 바로 사용 가능합니다!

### 사용 가능한 Scene

-   **lego** - 레고 불도저 🚜
-   **chair** - 의자 🪑
-   **drums** - 드럼 🥁
-   **ficus** - 화분 식물 🌿
-   **hotdog** - 핫도그 🌭
-   **materials** - 광택 구체들 ✨
-   **mic** - 마이크 🎤
-   **ship** - 해적선 ⛵

---

## 학습하기

### 방법 1: 명령줄에서 직접 실행

```bash
# Lego scene 학습 (30 에폭, 약 1-2시간)
python train_real.py lego

# 다른 scene 학습
python train_real.py chair
python train_real.py drums
```

### 방법 2: 스크립트 사용

```bash
# 학습 + 테스트 자동 실행
./run_real.sh lego
./run_real.sh chair
```

### 학습 파라미터

`config.py`에서 파라미터를 조정할 수 있습니다:

```python
TRAIN_CONFIG = {
    'n_epochs': 30,        # 에폭 수 (더 많이 = 더 좋은 품질)
    'batch_size': 1024,    # 배치 크기 (GPU 메모리에 맞게)
    'lr': 5e-4,            # Learning rate
    'N_samples': 64,       # Ray당 샘플 수
    'save_dir': './checkpoints',
    'log_interval': 5,     # 로그 출력 간격
}
```

---

## 테스트/렌더링

### 학습된 모델로 새로운 뷰 렌더링

```bash
# Test set으로 렌더링
python test_real.py lego

# Validation set으로 렌더링
python test_real.py lego val

# Training set으로 렌더링 (overfitting 확인)
python test_real.py lego train
```

### 결과물

렌더링 결과는 `./renders/{scene}/{split}/`에 저장됩니다:

-   `comparison_XXX.png` - Ground Truth vs 렌더링 vs 깊이 맵
-   `render_XXX.png` - 렌더링된 RGB 이미지만
-   `video/frame_XXX.png` - 360도 회전 비디오 프레임
-   `video_preview.png` - 비디오 프레임 미리보기

---

## 성능 지표

### PSNR (Peak Signal-to-Noise Ratio)

테스트 시 자동으로 계산됩니다:

-   **25-30 dB**: 학습 초기 단계
-   **30-35 dB**: 좋은 품질
-   **35+ dB**: 매우 좋은 품질

### 예상 학습 시간 (Mac M1/M2/M3)

| Scene | 이미지 수 | 30 에폭 | 50 에폭 |
| ----- | --------- | ------- | ------- |
| lego  | 100       | 1-2시간 | 2-3시간 |
| chair | 100       | 1-2시간 | 2-3시간 |
| drums | 100       | 1-2시간 | 2-3시간 |

---

## 문제 해결

### MPS 메모리 부족

```python
# config.py에서 배치 크기 감소
TRAIN_CONFIG = {
    'batch_size': 512,  # 기본값: 1024
}
```

### 학습이 너무 느림

```python
# 샘플 수 감소
TRAIN_CONFIG = {
    'N_samples': 32,  # 기본값: 64
}
```

### 품질이 낮음

-   에폭 수 증가: `n_epochs: 50` 또는 `100`
-   샘플 수 증가: `N_samples: 128`
-   Learning rate 조정: `lr: 1e-4`

---

## 고급 사용

### 체크포인트에서 이어서 학습

```bash
# 가장 최근 체크포인트에서 자동 재개
python resume_train.py lego

# 특정 체크포인트에서 재개
python resume_train.py lego nerf_epoch_20.pth

# 사용 가능한 체크포인트 확인
python resume_train.py lego list
```

자세한 내용은 `RESUME_GUIDE.md`를 참고하세요.

### 커스텀 카메라 경로

`test_real.py`의 `render_video()` 함수를 수정하여 원하는 카메라 경로를 만들 수 있습니다.

---

## 다음 단계

1. ✅ 빠른 데모 실행
2. ✅ Lego scene 학습
3. ✅ 테스트 및 렌더링
4. 🔄 다른 scene들 시도
5. 🔄 파라미터 튜닝
6. 🔄 결과 분석 및 비교

---

## 참고 자료

-   **원본 논문**: [NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934)
-   **공식 사이트**: [nerf-website](https://www.matthewtancik.com/nerf)
-   **Dataset**: [NeRF Synthetic Dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
