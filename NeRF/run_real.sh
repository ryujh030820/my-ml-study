#!/bin/bash

# 실제 NeRF synthetic dataset으로 학습 및 테스트 실행 스크립트

# Scene 선택 (기본값: lego)
SCENE=${1:-lego}

echo "===== NeRF 학습 및 테스트 (${SCENE}) ====="
echo ""

# 필요한 디렉토리 생성
mkdir -p checkpoints/${SCENE}
mkdir -p renders/${SCENE}

# 데이터셋 확인
if [ ! -d "./nerf_synthetic/${SCENE}" ]; then
    echo "오류: 데이터셋을 찾을 수 없습니다: ./nerf_synthetic/${SCENE}"
    echo "데이터셋을 다운로드하고 nerf_synthetic 폴더에 배치해주세요."
    echo ""
    echo "다운로드 링크:"
    echo "https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1"
    exit 1
fi

echo "1. 학습 시작..."
python train_real.py ${SCENE}

if [ $? -eq 0 ]; then
    echo ""
    echo "2. 학습 완료! 테스트 시작..."
    python test_real.py ${SCENE}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "===== 완료 ====="
        echo "체크포인트: ./checkpoints/${SCENE}/"
        echo "렌더링 결과: ./renders/${SCENE}/"
    else
        echo "테스트 중 오류 발생"
        exit 1
    fi
else
    echo "학습 중 오류 발생"
    exit 1
fi

