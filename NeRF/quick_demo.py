"""
간단한 NeRF 데모 스크립트
빠른 학습과 테스트를 위한 축소된 설정
"""
import torch
from model import NeRF, get_rays, render_rays
from data import get_simple_scene_data
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def quick_train_and_test():
    """빠른 학습 및 테스트"""
    
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 MPS(Metal) 가속 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 CUDA 가속 사용")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU 사용 (느릴 수 있습니다)")
    
    # 간단한 모델 생성
    print("\n📦 모델 생성 중...")
    model = NeRF(
        pos_L=6,  # 축소된 positional encoding
        view_L=2,
        hidden_dim=128,  # 작은 네트워크
        use_viewdirs=True
    )
    model = model.to(device)
    
    # 간단한 데이터 생성
    print("📊 합성 데이터 생성 중...")
    data = get_simple_scene_data(device=device, n_views=10)
    
    # 학습
    print("\n🎓 학습 시작...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 20
    batch_size = 512
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(data, desc=f"Epoch {epoch+1}/{n_epochs}")
        
        for view_data in pbar:
            H = view_data['H']
            W = view_data['W']
            focal = view_data['focal']
            c2w = view_data['c2w']
            target_img = view_data['image']
            
            # Ray 생성
            rays_o, rays_d = get_rays(H, W, focal, c2w)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            
            rays_o_flat = rays_o.reshape(-1, 3)
            rays_d_flat = rays_d.reshape(-1, 3)
            target_flat = target_img.reshape(-1, 3)
            
            # 랜덤 배치 선택
            n_rays = rays_o_flat.shape[0]
            indices = torch.randperm(n_rays, device=device)[:batch_size]
            
            rays_o_batch = rays_o_flat[indices]
            rays_d_batch = rays_d_flat[indices]
            target_batch = target_flat[indices]
            
            # 렌더링 및 학습
            rgb_pred, _ = render_rays(
                model, 
                rays_o_batch, 
                rays_d_batch,
                near=2.0, 
                far=6.0,
                N_samples=32,  # 적은 샘플 수
                device=device
            )
            
            loss = torch.nn.functional.mse_loss(rgb_pred, target_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} 평균 Loss: {avg_loss:.4f}")
    
    # 테스트 렌더링
    print("\n🎨 테스트 렌더링 중...")
    model.eval()
    
    # 새로운 뷰에서 렌더링
    test_view = data[0]
    H, W = test_view['H'], test_view['W']
    focal = test_view['focal']
    c2w = test_view['c2w']
    
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    rays_o = rays_o.to(device).reshape(-1, 3)
    rays_d = rays_d.to(device).reshape(-1, 3)
    
    # 청크 단위 렌더링
    chunk = 512
    rgb_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, rays_o.shape[0], chunk), desc="렌더링"):
            rgb, _ = render_rays(
                model,
                rays_o[i:i+chunk],
                rays_d[i:i+chunk],
                near=2.0,
                far=6.0,
                N_samples=32,
                device=device
            )
            rgb_list.append(rgb.cpu())
    
    rgb_map = torch.cat(rgb_list, dim=0).reshape(H, W, 3).numpy()
    target_map = test_view['image'].cpu().numpy()
    
    # 결과 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(target_map)
    axes[0].set_title('타겟 이미지')
    axes[0].axis('off')
    
    axes[1].imshow(rgb_map)
    axes[1].set_title('렌더링 결과')
    axes[1].axis('off')
    
    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('학습 Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_result.png', dpi=150, bbox_inches='tight')
    print("\n✅ 완료! 결과가 'demo_result.png'에 저장되었습니다.")
    plt.show()


if __name__ == '__main__':
    quick_train_and_test()

