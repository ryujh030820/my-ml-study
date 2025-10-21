# 체크포인트에서 학습 재개 가이드

## 📌 개요

학습 중 중단되었거나 추가 학습이 필요할 때 체크포인트에서 이어서 학습할 수 있습니다.

---

## 🚀 빠른 시작

### 방법 1: 자동으로 가장 최근 체크포인트에서 재개

```bash
python resume_train.py lego
```

### 방법 2: 특정 체크포인트 선택

```bash
python resume_train.py lego nerf_epoch_20.pth
```

### 방법 3: 수동으로 재개

```bash
python train_real.py lego nerf_epoch_20.pth
```

---

## 📋 사용 가능한 체크포인트 확인

```bash
# 체크포인트 목록 보기
python resume_train.py lego list

# 출력 예시:
# 사용 가능한 체크포인트 (./checkpoints/lego):
#   - nerf_epoch_5.pth (2.3 MB)
#   - nerf_epoch_10.pth (2.3 MB)
#   - nerf_epoch_15.pth (2.3 MB)
#   - nerf_epoch_20.pth (2.3 MB)
#   - nerf_final.pth (2.3 MB)
```

---

## 🔧 작동 원리

### 1. 저장되는 정보

각 체크포인트는 다음 정보를 포함합니다:

```python
{
    'epoch': 현재_에폭_번호,
    'model_state_dict': 모델_가중치,
    'optimizer_state_dict': optimizer_상태,
    'loss': 현재_loss
}
```

### 2. 재개 시 로드되는 것들

-   ✅ **모델 가중치**: 학습된 파라미터 복원
-   ✅ **Optimizer 상태**: learning rate, momentum 등 복원
-   ✅ **에폭 번호**: 저장된 다음 에폭부터 시작
-   ✅ **이전 loss**: 학습 진행 상황 확인

---

## 💡 사용 예시

### 예시 1: 30 에폭 학습 후 추가로 20 에폭 더 학습

```bash
# 1단계: 처음 30 에폭 학습
python train_real.py lego  # config.py에서 n_epochs=30

# 2단계: config.py 수정
# TRAIN_CONFIG = {
#     'n_epochs': 50,  # 30 → 50으로 변경
#     ...
# }

# 3단계: 체크포인트에서 재개 (30→50 에폭 학습됨)
python resume_train.py lego
```

### 예시 2: 특정 체크포인트에서 다시 시작

```bash
# 에폭 15 체크포인트에서 재개
python train_real.py lego nerf_epoch_15.pth
```

### 예시 3: 다른 scene들도 동일하게

```bash
python resume_train.py chair
python resume_train.py drums latest
python resume_train.py ficus nerf_epoch_10.pth
```

---

## 📊 체크포인트 관리

### 자동 저장

학습 중 자동으로 저장되는 체크포인트:

-   `nerf_epoch_5.pth` (log_interval=5 마다)
-   `nerf_epoch_10.pth`
-   `nerf_epoch_15.pth`
-   ...
-   `nerf_final.pth` (최종 모델)

### 수동 저장 간격 조정

`config.py`에서 조정:

```python
TRAIN_CONFIG = {
    'log_interval': 10,  # 10 에폭마다 저장
}
```

### 디스크 공간 관리

오래된 체크포인트 삭제:

```bash
# Lego scene의 오래된 체크포인트만 남기기
cd checkpoints/lego
ls -lt  # 목록 확인
rm nerf_epoch_5.pth nerf_epoch_10.pth  # 불필요한 것 삭제
```

---

## ⚠️ 주의사항

### 1. config.py 변경

체크포인트를 저장한 후 config.py를 변경하면:

-   ✅ `n_epochs` 변경: 안전 (더 많이 학습)
-   ✅ `batch_size`, `lr` 변경: 안전 (새로운 설정으로 학습)
-   ⚠️ 모델 구조 변경 (`MODEL_CONFIG`): 위험! (로드 실패 가능)

### 2. 에폭 범위

```python
# config.py
TRAIN_CONFIG = {
    'n_epochs': 50,
}

# 체크포인트: epoch=30에서 재개
# → 30부터 50까지 학습 (20 에폭 추가)
```

### 3. Optimizer 상태

-   Optimizer 상태가 없는 체크포인트: 새로운 optimizer로 학습
-   Learning rate 스케줄링을 사용한다면 주의 필요

---

## 🔍 문제 해결

### "체크포인트를 찾을 수 없습니다"

```bash
# 체크포인트 확인
python resume_train.py lego list

# 또는 직접 확인
ls -lh checkpoints/lego/
```

### "모델 가중치 로드 실패"

-   모델 구조가 변경되었을 수 있습니다
-   `model.py`의 변경사항을 되돌리세요

### Learning rate가 이상함

```python
# train_real.py에서 optimizer 새로 생성
optimizer_state = None  # optimizer state 무시
```

---

## 🎯 고급 사용

### 1. 특정 에폭부터 수동 재개

```python
# custom_resume.py
import torch
from train_real import train, main

# 체크포인트 로드
checkpoint = torch.load('checkpoints/lego/nerf_epoch_20.pth')

# 모델 생성 및 로드
model = NeRF(...)
model.load_state_dict(checkpoint['model_state_dict'])

# 특정 에폭부터 학습
train(model, data, device, start_epoch=20, n_epochs=100)
```

### 2. Learning rate 변경하면서 재개

```python
# config.py 수정 후
# 더 낮은 learning rate로 fine-tuning
TRAIN_CONFIG = {
    'lr': 1e-4,  # 5e-4 → 1e-4
    'n_epochs': 60,
}

# 재개
python resume_train.py lego
```

### 3. 여러 scene 일괄 재개

```bash
#!/bin/bash
for scene in lego chair drums ficus; do
    echo "=== $scene ==="
    python resume_train.py $scene
done
```

---

## 📚 관련 문서

-   `USAGE.md` - 전체 사용 가이드
-   `README.md` - 프로젝트 개요
-   `config.py` - 학습 설정 파일
