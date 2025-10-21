"""
실제 NeRF synthetic dataset으로 학습된 모델 테스트
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from model import NeRF, get_rays, render_rays
from data import SyntheticDataset
from config import DATASET_CONFIG, MODEL_CONFIG, RENDER_CONFIG


def render_image(model, H, W, focal, c2w, device, near, far, N_samples=128, chunk=1024):
    """전체 이미지 렌더링"""
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
        for i in tqdm(range(0, n_rays, chunk), desc="렌더링", leave=False):
            rays_o_batch = rays_o_flat[i:i+chunk]
            rays_d_batch = rays_d_flat[i:i+chunk]
            
            rgb, depth = render_rays(
                model,
                rays_o_batch,
                rays_d_batch,
                near=near,
                far=far,
                N_samples=N_samples,
                device=device
            )
            
            rgb_list.append(rgb.cpu())
            depth_list.append(depth.cpu())
    
    rgb_map = torch.cat(rgb_list, dim=0).reshape(H, W, 3)
    depth_map = torch.cat(depth_list, dim=0).reshape(H, W)
    
    return rgb_map.numpy(), depth_map.numpy()


def test_model(scene, split='test', checkpoint='nerf_final.pth'):
    """
    학습된 모델로 테스트 셋 렌더링
    Args:
        scene: scene 이름 (lego, chair, etc.)
        split: 'test', 'val', 또는 'train'
        checkpoint: 체크포인트 파일명
    """
    if scene not in DATASET_CONFIG:
        print(f"오류: '{scene}'는 지원하지 않는 scene입니다.")
        print(f"사용 가능한 scene: {list(DATASET_CONFIG.keys())}")
        return
    
    print("=" * 60)
    print(f"NeRF 테스트 - {scene.upper()} Scene ({split})")
    print("=" * 60)
    
    # Scene 설정
    scene_config = DATASET_CONFIG[scene]
    datadir = scene_config['datadir']
    near = scene_config['near']
    far = scene_config['far']
    radius = scene_config['render_radius']
    
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 MPS 사용")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 CUDA 사용")
    else:
        device = torch.device("cpu")
        print("⚠️  CPU 사용")
    
    # 모델 로드
    model = NeRF(**MODEL_CONFIG)
    checkpoint_path = f'./checkpoints/{scene}/{checkpoint}'
    
    if not os.path.exists(checkpoint_path):
        print(f"\n오류: 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        print(f"먼저 학습을 실행하세요: python train_real.py {scene}")
        return
    
    print(f"\n체크포인트 로드: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 테스트 데이터 로드
    if not os.path.exists(datadir):
        print(f"\n오류: 데이터셋을 찾을 수 없습니다: {datadir}")
        return
    
    print(f"데이터셋 로드: {datadir} ({split})")
    dataset = SyntheticDataset(datadir, split=split, half_res=True, white_bkgd=True)
    print(f"데이터셋 크기: {len(dataset)} 이미지")
    
    # 렌더링 결과 저장 디렉토리
    save_dir = f'./renders/{scene}/{split}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 몇 개 샘플 렌더링
    n_samples = min(5, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    
    print(f"\n{n_samples}개 이미지 렌더링 중...")
    
    psnr_list = []
    
    for idx in indices:
        view_data = dataset[idx]
        H, W = view_data['H'], view_data['W']
        focal = view_data['focal']
        c2w = view_data['c2w'].to(device)
        target = view_data['image'].numpy()
        
        print(f"\n이미지 {idx} 렌더링...")
        rgb_map, depth_map = render_image(
            model, H, W, focal, c2w, device, near, far,
            N_samples=RENDER_CONFIG['N_samples'],
            chunk=RENDER_CONFIG['chunk']
        )
        
        # PSNR 계산
        mse = np.mean((rgb_map - target) ** 2)
        psnr = -10 * np.log10(mse)
        psnr_list.append(psnr)
        
        # 결과 저장
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(target)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        axes[1].imshow(rgb_map)
        axes[1].set_title(f'Rendered (PSNR: {psnr:.2f})')
        axes[1].axis('off')
        
        axes[2].imshow(depth_map, cmap='viridis')
        axes[2].set_title('Depth Map')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'comparison_{idx:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # RGB 이미지 별도 저장
        img = (np.clip(rgb_map, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(save_dir, f'render_{idx:03d}.png'))
        
        print(f"  저장: {save_path} (PSNR: {psnr:.2f} dB)")
    
    avg_psnr = np.mean(psnr_list)
    print("\n" + "=" * 60)
    print(f"평균 PSNR: {avg_psnr:.2f} dB")
    print(f"결과 저장 위치: {save_dir}")
    print("=" * 60)
    
    # 360도 비디오 렌더링
    print("\n360도 회전 비디오 렌더링 중...")
    render_video(model, scene, device, near, far, radius, save_dir)


def render_video(model, scene, device, near, far, radius, save_dir, n_frames=30):
    """360도 회전 비디오 렌더링"""
    video_dir = os.path.join(save_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)
    
    H, W = 200, 200
    focal = 200.0
    
    frames = []
    
    for i in tqdm(range(n_frames), desc="비디오 프레임 렌더링"):
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
        rgb_map, _ = render_image(
            model, H, W, focal, c2w, device, near, far,
            N_samples=64, chunk=1024
        )
        
        frames.append(rgb_map)
        
        # 프레임 저장
        img = (np.clip(rgb_map, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(video_dir, f'frame_{i:03d}.png'))
    
    print(f"비디오 프레임 저장: {video_dir}")
    
    # 그리드로 시각화
    n_show = min(8, len(frames))
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
    
    _, axes = plt.subplots(2, n_show // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        axes[i].imshow(frames[idx])
        axes[i].axis('off')
        axes[i].set_title(f'Frame {idx}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'video_preview.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    # 명령줄 인자
    scene = 'lego'
    split = 'test'
    checkpoint = 'nerf_final.pth'
    
    if len(sys.argv) > 1:
        scene = sys.argv[1]
    if len(sys.argv) > 2:
        split = sys.argv[2]
    if len(sys.argv) > 3:
        checkpoint = sys.argv[3]
    
    test_model(scene, split, checkpoint)


if __name__ == '__main__':
    main()

