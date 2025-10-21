import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import NeRF, get_rays, render_rays
from data import SyntheticDataset, get_simple_scene_data


def train(
    model,
    data,
    device,
    n_epochs=100,
    batch_size=1024,
    lr=5e-4,
    N_samples=64,
    save_dir='./checkpoints',
    log_interval=10,
    start_epoch=0,
    optimizer_state=None
):
    """
    NeRF 모델 학습
    Args:
        model: NeRF 모델
        data: 학습 데이터 리스트
        device: 'mps', 'cuda', 또는 'cpu'
        n_epochs: 에폭 수
        batch_size: 배치당 ray 수
        lr: learning rate
        N_samples: ray당 샘플 수
        save_dir: 체크포인트 저장 디렉토리
        log_interval: 로그 출력 간격
        start_epoch: 시작 에폭 (체크포인트에서 재개 시)
        optimizer_state: optimizer state dict (체크포인트에서 재개 시)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # optimizer state 로드
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Optimizer state 로드 완료")
        except Exception as e:
            print(f"경고: Optimizer state 로드 실패: {e}")
    
    criterion = nn.MSELoss()
    
    train_losses = []
    
    print(f"학습 시작: device={device}, epochs={start_epoch}→{n_epochs}, batch_size={batch_size}")
    print(f"데이터 뷰 수: {len(data)}")
    
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
                    near=2.0, 
                    far=6.0,
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
    print(f"최종 모델 저장: {final_path}")
    
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
    # 디바이스 설정 (Mac에서 MPS 사용)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS(Metal Performance Shaders) 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA 사용")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    
    # 데이터 로드
    datadir = './nerf_synthetic/lego'  # 실제 NeRF synthetic dataset 경로
    
    if os.path.exists(datadir):
        print(f"데이터셋 로드: {datadir}")
        dataset = SyntheticDataset(datadir, split='train', half_res=True)
        print(f"데이터셋 크기: {len(dataset)} 이미지")
        # DataLoader 대신 직접 접근
        data = [dataset[i] for i in range(len(dataset))]
    else:
        print("실제 데이터셋이 없어 간단한 합성 데이터를 생성합니다.")
        print(f"경로를 확인하세요: {datadir}")
        data = get_simple_scene_data(device=device, n_views=20)
    
    # 모델 초기화
    model = NeRF(
        pos_L=10,
        view_L=4,
        hidden_dim=256,
        use_viewdirs=True
    )
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 학습
    model, _ = train(
        model=model,
        data=data,
        device=device,
        n_epochs=30,  # 실제 데이터셋은 100장이므로 30 에폭으로 충분
        batch_size=1024,
        lr=5e-4,
        N_samples=64,
        save_dir='./checkpoints',
        log_interval=5
    )
    
    print("학습 완료!")


if __name__ == '__main__':
    main()

