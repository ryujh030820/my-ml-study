"""
체크포인트에서 학습 재개를 위한 편의 스크립트
"""
import os
import sys
import glob


def find_latest_checkpoint(checkpoint_dir):
    """가장 최근 체크포인트 찾기"""
    pattern = os.path.join(checkpoint_dir, 'nerf_epoch_*.pth')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # 에폭 번호로 정렬
    def get_epoch(path):
        basename = os.path.basename(path)
        # nerf_epoch_30.pth -> 30
        try:
            return int(basename.split('_')[-1].split('.')[0])
        except:
            return 0
    
    checkpoints.sort(key=get_epoch)
    return checkpoints[-1]


def list_checkpoints(checkpoint_dir):
    """모든 체크포인트 나열"""
    if not os.path.exists(checkpoint_dir):
        return []
    
    pattern = os.path.join(checkpoint_dir, 'nerf_epoch_*.pth')
    checkpoints = glob.glob(pattern)
    
    # 최종 모델도 포함
    final_path = os.path.join(checkpoint_dir, 'nerf_final.pth')
    if os.path.exists(final_path):
        checkpoints.append(final_path)
    
    return sorted(checkpoints)


def main():
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python resume_train.py <scene> [checkpoint]")
        print()
        print("예시:")
        print("  python resume_train.py lego                    # 가장 최근 체크포인트에서 재개")
        print("  python resume_train.py lego nerf_epoch_20.pth  # 특정 체크포인트에서 재개")
        print("  python resume_train.py lego latest             # 가장 최근 체크포인트 명시적으로")
        print("  python resume_train.py lego list               # 사용 가능한 체크포인트 목록")
        sys.exit(1)
    
    scene = sys.argv[1]
    checkpoint_dir = f'./checkpoints/{scene}'
    
    # 체크포인트 디렉토리 확인
    if not os.path.exists(checkpoint_dir):
        print(f"오류: 체크포인트 디렉토리를 찾을 수 없습니다: {checkpoint_dir}")
        print(f"먼저 '{scene}' scene을 학습해주세요.")
        sys.exit(1)
    
    # list 명령
    if len(sys.argv) > 2 and sys.argv[2] == 'list':
        checkpoints = list_checkpoints(checkpoint_dir)
        if not checkpoints:
            print(f"체크포인트가 없습니다: {checkpoint_dir}")
        else:
            print(f"\n사용 가능한 체크포인트 ({checkpoint_dir}):")
            for cp in checkpoints:
                basename = os.path.basename(cp)
                size = os.path.getsize(cp) / (1024 * 1024)
                print(f"  - {basename} ({size:.1f} MB)")
        sys.exit(0)
    
    # 체크포인트 선택
    if len(sys.argv) > 2 and sys.argv[2] != 'latest':
        checkpoint = sys.argv[2]
    else:
        # 가장 최근 체크포인트 찾기
        checkpoint = find_latest_checkpoint(checkpoint_dir)
        if checkpoint:
            checkpoint = os.path.basename(checkpoint)
        else:
            print(f"오류: 체크포인트를 찾을 수 없습니다: {checkpoint_dir}")
            print("사용 가능한 체크포인트:")
            os.system(f"python resume_train.py {scene} list")
            sys.exit(1)
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
    
    if not os.path.exists(checkpoint_path):
        print(f"오류: 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print("\n사용 가능한 체크포인트:")
        checkpoints = list_checkpoints(checkpoint_dir)
        for cp in checkpoints:
            print(f"  - {os.path.basename(cp)}")
        sys.exit(1)
    
    # 학습 재개
    print("=" * 60)
    print(f"학습 재개: {scene}")
    print(f"체크포인트: {checkpoint}")
    print("=" * 60)
    print()
    
    cmd = f"python train_real.py {scene} {checkpoint}"
    print(f"실행: {cmd}\n")
    os.system(cmd)


if __name__ == '__main__':
    main()

