#!/bin/bash

# NeRF 학습 및 테스트 실행 스크립트

echo "===== NeRF 학습 및 테스트 ====="
echo ""

# 필요한 디렉토리 생성
mkdir -p checkpoints
mkdir -p renders
mkdir -p data

echo "1. 학습 시작..."
python train.py

if [ $? -eq 0 ]; then
    echo ""
    echo "2. 학습 완료! 테스트 시작..."
    python test.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "===== 완료 ====="
        echo "체크포인트: ./checkpoints/"
        echo "렌더링 결과: ./renders/"
    else
        echo "테스트 중 오류 발생"
        exit 1
    fi
else
    echo "학습 중 오류 발생"
    exit 1
fi

