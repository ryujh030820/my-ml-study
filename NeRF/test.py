import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from model import NeRF, get_rays, render_rays


def render_image(model, H, W, focal, c2w, device, N_samples=64, chunk=1024):
    """
    전체 이미지 렌더링
    Args:
        model: NeRF 모델
        H, W: 이미지 높이, 너비
        focal: 초점 거리
        c2w: camera-to-world 변환 행렬
        device: 디바이스
        N_samples: ray당 샘플 수
        chunk: 한번에 처리할 ray 수
    Returns:
        rgb_map: (H, W, 3) 렌더링된 이미지
        depth_map: (H, W) 깊이 맵
    """
    model.eval()
    
    rays_o, rays_d = get_rays(H, W, focal, c2w)
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)
    
    n_rays = rays_o_flat.shape[0]
    
    rgb_list = []
    depth_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, n_rays, chunk), desc="Rendering"):
            rays_o_batch = rays_o_flat[i:i+chunk]
            rays_d_batch = rays_d_flat[i:i+chunk]
            
            rgb, depth = render_rays(
                model,
                rays_o_batch,
                rays_d_batch,
                near=2.0,
                far=6.0,
                N_samples=N_samples,
                device=device
            )
            
            rgb_list.append(rgb.cpu())
            depth_list.append(depth.cpu())
    
    rgb_map = torch.cat(rgb_list, dim=0).reshape(H, W, 3)
    depth_map = torch.cat(depth_list, dim=0).reshape(H, W)
    
    return rgb_map.numpy(), depth_map.numpy()


def render_video(model, n_frames, device, save_dir='./renders', H=200, W=200, focal=200.0, radius=4.0):
    """
    회전하는 카메라로 비디오 렌더링
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    frames = []
    
    for i in tqdm(range(n_frames), desc="Rendering video frames"):
        angle = 2 * np.pi * i / n_frames
        
        # 카메라 위치
        cam_pos = np.array([
            radius * np.cos(angle),
            0.0,
            radius * np.sin(angle)
        ])
        
        # 카메라가 원점을 바라보도록
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = forward
        c2w[:3, 3] = cam_pos
        
        c2w = torch.from_numpy(c2w).to(device)
        
        # 렌더링
        rgb_map, _ = render_image(model, H, W, focal, c2w, device, N_samples=64, chunk=1024)
        
        frames.append(rgb_map)
        
        # 개별 프레임 저장
        img = (rgb_map * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_dir, f'frame_{i:03d}.png'))
    
    return frames


def visualize_results(frames, save_path='./renders/results.png'):
    """
    렌더링 결과 시각화
    """
    n_frames = len(frames)
    n_show = min(8, n_frames)
    
    _, axes = plt.subplots(2, n_show // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    indices = np.linspace(0, n_frames - 1, n_show, dtype=int)
    
    for i, idx in enumerate(indices):
        axes[i].imshow(frames[idx])
        axes[i].axis('off')
        axes[i].set_title(f'Frame {idx}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"결과 시각화 저장: {save_path}")


def test_single_view(model, device, save_dir='./renders'):
    """
    단일 뷰 테스트 렌더링
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 테스트 카메라 설정
    H, W = 200, 200
    focal = 200.0
    
    # 카메라 위치
    cam_pos = np.array([4.0, 1.0, 0.0])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0, 1, 0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos
    
    c2w = torch.from_numpy(c2w).to(device)
    
    # 렌더링
    print("테스트 이미지 렌더링 중...")
    rgb_map, depth_map = render_image(model, H, W, focal, c2w, device, N_samples=64, chunk=1024)
    
    # 결과 저장
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(rgb_map)
    axes[0].set_title('Rendered RGB')
    axes[0].axis('off')
    
    axes[1].imshow(depth_map, cmap='viridis')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'test_render.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"테스트 렌더링 저장: {save_path}")
    
    # RGB 이미지 별도 저장
    img = (rgb_map * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(save_dir, 'test_rgb.png'))
    
    return rgb_map, depth_map


def main():
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS(Metal Performance Shaders) 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA 사용")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    
    # 모델 로드
    model = NeRF(
        pos_L=10,
        view_L=4,
        hidden_dim=256,
        use_viewdirs=True
    )
    
    checkpoint_path = './checkpoints/nerf_final.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"체크포인트 로드: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"경고: 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print("학습되지 않은 모델로 테스트를 진행합니다.")
    
    model = model.to(device)
    model.eval()
    
    # 단일 뷰 테스트
    print("\n=== 단일 뷰 렌더링 ===")
    test_single_view(model, device, save_dir='./renders')
    
    # 비디오 렌더링
    print("\n=== 360도 회전 비디오 렌더링 ===")
    n_frames = 30
    frames = render_video(
        model, 
        n_frames=n_frames, 
        device=device, 
        save_dir='./renders',
        H=200, 
        W=200, 
        focal=200.0
    )
    
    # 결과 시각화
    visualize_results(frames, save_path='./renders/results.png')
    
    print("\n렌더링 완료!")
    print("결과 저장 위치: ./renders/")


if __name__ == '__main__':
    main()

