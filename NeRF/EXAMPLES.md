# NeRF 사용 예제 모음

## 📌 기본 워크플로우

### 예제 1: 처음부터 끝까지 (Lego)

```bash
# 1. 학습 (30 에폭, 약 1-2시간)
python train_real.py lego

# 2. 테스트
python test_real.py lego

# 3. 결과 확인
open renders/lego/test/comparison_000.png
open renders/lego/test/video_preview.png
```

---

## 🔄 체크포인트 재개 예제

### 예제 2: 학습 중단 후 재개

```bash
# 1. 처음 20 에폭 학습 중 Ctrl+C로 중단
python train_real.py lego

# 2. 체크포인트 확인
python resume_train.py lego list

# 출력:
# - nerf_epoch_5.pth
# - nerf_epoch_10.pth
# - nerf_epoch_15.pth
# - nerf_epoch_20.pth

# 3. 가장 최근 체크포인트에서 재개
python resume_train.py lego

# 또는 특정 체크포인트 선택
python resume_train.py lego nerf_epoch_15.pth
```

### 예제 3: 추가 학습

```bash
# 1. 처음 30 에폭 학습 완료
python train_real.py lego

# 2. config.py 수정 (30 → 50 에폭으로 변경)
# TRAIN_CONFIG = {
#     'n_epochs': 50,  # 변경
#     ...
# }

# 3. 추가 20 에폭 학습 (30→50)
python resume_train.py lego

# 4. 다시 config.py 수정 (50 → 100)
# TRAIN_CONFIG = {
#     'n_epochs': 100,
#     ...
# }

# 5. 추가 50 에폭 학습 (50→100)
python resume_train.py lego
```

### 예제 4: Fine-tuning (낮은 learning rate)

```bash
# 1. 처음 30 에폭 학습 완료
python train_real.py lego

# 2. config.py 수정
# TRAIN_CONFIG = {
#     'n_epochs': 50,
#     'lr': 1e-4,  # 5e-4 → 1e-4 (더 낮은 learning rate)
#     ...
# }

# 3. Fine-tuning (30→50 에폭)
python resume_train.py lego
```

---

## 🎨 다양한 Scene 학습

### 예제 5: 여러 Scene 순차 학습

```bash
# 각 scene별로 학습
python train_real.py lego
python train_real.py chair
python train_real.py drums
python train_real.py ficus

# 모두 테스트
python test_real.py lego
python test_real.py chair
python test_real.py drums
python test_real.py ficus
```

### 예제 6: 배치 스크립트로 자동화

```bash
# batch_train.sh 생성
cat > batch_train.sh << 'EOF'
#!/bin/bash
scenes=("lego" "chair" "drums" "ficus")

for scene in "${scenes[@]}"; do
    echo "=== Training $scene ==="
    python train_real.py $scene

    echo "=== Testing $scene ==="
    python test_real.py $scene
done
EOF

chmod +x batch_train.sh
./batch_train.sh
```

---

## 🔧 설정 조정 예제

### 예제 7: 빠른 프로토타입 (낮은 품질)

```python
# config.py 수정
TRAIN_CONFIG = {
    'n_epochs': 10,        # 30 → 10
    'batch_size': 2048,    # 1024 → 2048
    'N_samples': 32,       # 64 → 32
    'lr': 1e-3,            # 5e-4 → 1e-3
}
```

```bash
python train_real.py lego  # 약 20-30분
```

### 예제 8: 고품질 학습 (느리지만 좋은 품질)

```python
# config.py 수정
TRAIN_CONFIG = {
    'n_epochs': 100,       # 30 → 100
    'batch_size': 512,     # 1024 → 512
    'N_samples': 128,      # 64 → 128
    'lr': 5e-4,
}

RENDER_CONFIG = {
    'N_samples': 256,      # 128 → 256
}
```

```bash
python train_real.py lego  # 약 5-8시간
python test_real.py lego   # 고품질 렌더링
```

---

## 📊 결과 비교

### 예제 9: Train/Val/Test 모두 렌더링

```bash
# Lego scene의 모든 split 렌더링
python test_real.py lego train
python test_real.py lego val
python test_real.py lego test

# 결과 위치:
# renders/lego/train/
# renders/lego/val/
# renders/lego/test/
```

### 예제 10: 여러 에폭의 결과 비교

```bash
# 에폭 10 결과
python test_real.py lego test nerf_epoch_10.pth

# 결과를 다른 이름으로 복사
mv renders/lego/test renders/lego/test_epoch10

# 에폭 20 결과
python test_real.py lego test nerf_epoch_20.pth
mv renders/lego/test renders/lego/test_epoch20

# 최종 결과
python test_real.py lego test nerf_final.pth
mv renders/lego/test renders/lego/test_final

# 결과 비교
open renders/lego/test_epoch10/comparison_000.png
open renders/lego/test_epoch20/comparison_000.png
open renders/lego/test_final/comparison_000.png
```

---

## 💾 체크포인트 관리

### 예제 11: 디스크 공간 절약

```bash
# 체크포인트 확인
ls -lh checkpoints/lego/

# 중간 체크포인트 삭제 (최종 모델만 유지)
cd checkpoints/lego
rm nerf_epoch_*.pth
ls -lh  # nerf_final.pth만 남음
```

### 예제 12: 백업

```bash
# 중요한 체크포인트 백업
cp checkpoints/lego/nerf_final.pth backups/lego_final_30epochs.pth

# config 수정 후 추가 학습
# (원본은 백업되어 있음)
```

---

## 🚀 고급 워크플로우

### 예제 13: 2단계 학습

```bash
# 1단계: 빠르게 학습 (높은 learning rate)
# config.py: lr=1e-3, n_epochs=20
python train_real.py lego

# 2단계: Fine-tuning (낮은 learning rate)
# config.py: lr=1e-4, n_epochs=50
python resume_train.py lego

# 결과 확인
python test_real.py lego
```

### 예제 14: 실험 추적

```bash
# 실험 1: 기본 설정
python train_real.py lego
python test_real.py lego
mv checkpoints/lego checkpoints/lego_exp1_baseline
mv renders/lego renders/lego_exp1_baseline

# 실험 2: 높은 learning rate
# config.py: lr=1e-3
python train_real.py lego
python test_real.py lego
mv checkpoints/lego checkpoints/lego_exp2_high_lr
mv renders/lego renders/lego_exp2_high_lr

# 실험 3: 더 많은 샘플
# config.py: N_samples=128
python train_real.py lego
python test_real.py lego
mv checkpoints/lego checkpoints/lego_exp3_more_samples
mv renders/lego renders/lego_exp3_more_samples

# 결과 비교
open renders/lego_exp*/test/comparison_000.png
```

---

## 🎯 빠른 참조

### 자주 사용하는 명령어

```bash
# 학습
python train_real.py lego

# 체크포인트 확인
python resume_train.py lego list

# 재개
python resume_train.py lego

# 테스트
python test_real.py lego

# 전체 파이프라인
./run_real.sh lego
```

### 디렉토리 구조

```
checkpoints/lego/
├── nerf_epoch_5.pth
├── nerf_epoch_10.pth
└── nerf_final.pth

renders/lego/test/
├── comparison_000.png  # GT vs 렌더링 vs 깊이
├── render_000.png      # 렌더링만
├── video/
│   ├── frame_000.png   # 360도 비디오
│   └── ...
└── video_preview.png   # 비디오 미리보기
```

---

## 📚 더 알아보기

-   `README.md` - 프로젝트 개요
-   `USAGE.md` - 전체 사용 가이드
-   `RESUME_GUIDE.md` - 체크포인트 재개 상세 가이드
-   `config.py` - 설정 파일
