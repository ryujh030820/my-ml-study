"""
실제 NeRF synthetic dataset으로 학습하는 스크립트
config.py를 사용하여 다양한 scene을 쉽게 학습 가능
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import NeRF, get_rays, render_rays
from data import SyntheticDataset
from config import DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG


def train(
    model,
    data,
    device,
    near,
    far,
    n_epochs=30,
    batch_size=1024,
    lr=5e-4,
    N_samples=64,
    save_dir='./checkpoints',
    log_interval=5,
    start_epoch=0,
    optimizer_state=None
):
    """NeRF 모델 학습"""
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # optimizer state 로드 (체크포인트에서 재개하는 경우)
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Optimizer state 로드 완료")
        except Exception as e:
            print(f"경고: Optimizer state 로드 실패: {e}")
            print("새로운 optimizer로 학습을 시작합니다.")
    
    criterion = nn.MSELoss()
    
    train_losses = []
    
    print(f"\n학습 시작:")
    print(f"  - Device: {device}")
    print(f"  - Epochs: {start_epoch} → {n_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {lr}")
    print(f"  - 데이터 뷰 수: {len(data)}")
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # 각 뷰에 대해 학습
        pbar = tqdm(enumerate(data), total=len(data), desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for _, view_data in pbar:
            H = view_data['H']
            W = view_data['W']
            focal = view_data['focal']
            c2w = view_data['c2w'].to(device)
            target_img = view_data['image'].to(device)
            
            # Ray 생성
            rays_o, rays_d = get_rays(H, W, focal, c2w)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            
            # 배치 단위로 처리
            rays_o_flat = rays_o.reshape(-1, 3)
            rays_d_flat = rays_d.reshape(-1, 3)
            target_flat = target_img.reshape(-1, 3)
            
            n_rays = rays_o_flat.shape[0]
            indices = torch.randperm(n_rays, device=device)
            
            view_loss = 0.0
            view_batches = 0
            
            for i in range(0, n_rays, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                rays_o_batch = rays_o_flat[batch_indices]
                rays_d_batch = rays_d_flat[batch_indices]
                target_batch = target_flat[batch_indices]
                
                # 렌더링
                rgb_pred, _ = render_rays(
                    model, 
                    rays_o_batch, 
                    rays_d_batch,
                    near=near,
                    far=far,
                    N_samples=N_samples,
                    device=device
                )
                
                # Loss 계산 및 역전파
                loss = criterion(rgb_pred, target_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                view_loss += loss.item()
                view_batches += 1
            
            avg_view_loss = view_loss / view_batches if view_batches > 0 else 0
            epoch_loss += avg_view_loss
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{avg_view_loss:.4f}'})
        
        avg_epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0
        train_losses.append(avg_epoch_loss)
        
        # 로그 출력
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_epoch_loss:.4f}")
            
            # 체크포인트 저장
            checkpoint_path = os.path.join(save_dir, f'nerf_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"체크포인트 저장: {checkpoint_path}")
    
    # 최종 모델 저장
    final_path = os.path.join(save_dir, 'nerf_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
    }, final_path)
    print(f"\n최종 모델 저장: {final_path}")
    
    # Loss 그래프 저장
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    plt.close()
    print(f"학습 loss 그래프 저장: {os.path.join(save_dir, 'training_loss.png')}")
    
    return model, train_losses


def main():
    # 명령줄 인자 파싱
    scene = 'lego'
    resume_checkpoint = None
    
    if len(sys.argv) > 1:
        scene = sys.argv[1]
    if len(sys.argv) > 2:
        resume_checkpoint = sys.argv[2]
    
    if scene not in DATASET_CONFIG:
        print(f"오류: '{scene}'는 지원하지 않는 scene입니다.")
        print(f"사용 가능한 scene: {list(DATASET_CONFIG.keys())}")
        sys.exit(1)
    
    print("=" * 60)
    print(f"NeRF 학습 - {scene.upper()} Scene")
    print("=" * 60)
    
    # Scene 설정 로드
    scene_config = DATASET_CONFIG[scene]
    datadir = scene_config['datadir']
    near = scene_config['near']
    far = scene_config['far']
    
    # 디바이스 설정 (Mac에서 MPS 사용)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 MPS(Metal Performance Shaders) 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 CUDA 사용")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU 사용")
    
    # 데이터 로드
    if not os.path.exists(datadir):
        print(f"\n오류: 데이터셋을 찾을 수 없습니다: {datadir}")
        print("nerf_synthetic 폴더가 있는지 확인해주세요.")
        sys.exit(1)
    
    print(f"\n데이터셋 로드: {datadir}")
    dataset = SyntheticDataset(datadir, split='train', half_res=True, white_bkgd=True)
    print(f"데이터셋 크기: {len(dataset)} 이미지")
    
    # 데이터 리스트 생성
    data = [dataset[i] for i in range(len(dataset))]
    
    # 이미지 크기 확인
    sample_data = data[0]
    print(f"이미지 크기: {sample_data['H']} x {sample_data['W']}")
    print(f"초점 거리: {sample_data['focal']:.2f}")
    
    # 모델 초기화
    print("\n모델 초기화...")
    model = NeRF(**MODEL_CONFIG)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 체크포인트 디렉토리 설정
    checkpoint_dir = f'./checkpoints/{scene}'
    
    # 체크포인트에서 재개
    start_epoch = 0
    optimizer = None
    
    if resume_checkpoint:
        checkpoint_path = resume_checkpoint
        if not os.path.isabs(checkpoint_path):
            # 상대 경로인 경우 체크포인트 디렉토리 기준
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
        
        if os.path.exists(checkpoint_path):
            print(f"\n체크포인트에서 재개: {checkpoint_path}")
            checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # optimizer state 로드 준비
            if 'optimizer_state_dict' in checkpoint_data:
                optimizer_state = checkpoint_data['optimizer_state_dict']
            else:
                optimizer_state = None
            
            # 시작 에폭 설정
            if 'epoch' in checkpoint_data:
                start_epoch = checkpoint_data['epoch']
                print(f"에폭 {start_epoch}부터 재개합니다.")
            
            # 이전 loss 정보
            if 'loss' in checkpoint_data:
                print(f"이전 학습 loss: {checkpoint_data['loss']:.4f}")
        else:
            print(f"\n경고: 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
            print("새로 학습을 시작합니다.")
            optimizer_state = None
    else:
        optimizer_state = None
    
    # 학습 설정 복사 및 수정
    train_config = TRAIN_CONFIG.copy()
    train_config['save_dir'] = checkpoint_dir
    
    # 학습
    print("\n" + "=" * 60)
    model, _ = train(
        model=model,
        data=data,
        device=device,
        near=near,
        far=far,
        start_epoch=start_epoch,
        optimizer_state=optimizer_state,
        **train_config
    )
    
    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)
    print(f"체크포인트 위치: {checkpoint_dir}")
    print(f"\n테스트 실행: python test_real.py {scene}")
    print(f"이어서 학습: python train_real.py {scene} nerf_final.pth")


if __name__ == '__main__':
    main()

