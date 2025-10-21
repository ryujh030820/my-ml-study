import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SyntheticDataset(Dataset):
    """
    Blender synthetic dataset (NeRF 논문에서 사용)
    간단한 합성 데이터셋
    """
    def __init__(self, datadir, split='train', half_res=True, white_bkgd=True):
        self.datadir = datadir
        self.split = split
        self.half_res = half_res
        self.white_bkgd = white_bkgd
        
        # transforms_{split}.json 파일 로드
        transforms_file = os.path.join(datadir, f'transforms_{split}.json')
        
        if os.path.exists(transforms_file):
            with open(transforms_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            self.camera_angle_x = meta['camera_angle_x']
            self.frames = meta['frames']
        else:
            # 데이터셋이 없으면 간단한 예제 데이터 생성
            print(f"Warning: {transforms_file} not found. Creating synthetic data...")
            self.create_simple_dataset()
    
    def create_simple_dataset(self):
        """간단한 합성 데이터 생성 (테스트용)"""
        self.camera_angle_x = 0.6911112070083618
        self.frames = []
        
        # 간단한 원형 카메라 경로
        n_frames = 20 if self.split == 'train' else 5
        for i in range(n_frames):
            angle = 2 * np.pi * i / n_frames
            radius = 4.0
            
            # Camera-to-world 변환 행렬
            transform_matrix = [
                [np.cos(angle), 0, np.sin(angle), radius * np.sin(angle)],
                [0, 1, 0, 0],
                [-np.sin(angle), 0, np.cos(angle), radius * np.cos(angle)],
                [0, 0, 0, 1]
            ]
            
            self.frames.append({
                'file_path': f'./dummy_{i}',
                'transform_matrix': transform_matrix
            })
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # 이미지 로드
        img_path = os.path.join(self.datadir, frame['file_path'])
        if not img_path.endswith('.png'):
            img_path += '.png'
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = np.array(img) / 255.0
            
            if self.half_res:
                H, W = img.shape[:2]
                img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(
                    (W // 2, H // 2), Image.LANCZOS
                )) / 255.0
            
            # 알파 채널 처리 (흰색 배경)
            if img.shape[-1] == 4:
                if self.white_bkgd:
                    img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])
                else:
                    img = img[..., :3]
            
            H, W = img.shape[:2]
        else:
            # 더미 이미지 생성
            H, W = 400, 400
            if self.half_res:
                H, W = 200, 200
            img = np.ones((H, W, 3)) * 0.5  # 회색 이미지
        
        # 카메라 파라미터
        focal = 0.5 * W / np.tan(0.5 * self.camera_angle_x)
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        
        return {
            'image': torch.from_numpy(img).float(),
            'c2w': torch.from_numpy(c2w).float(),
            'focal': focal,
            'H': H,
            'W': W
        }


def get_simple_scene_data(device='cpu', n_views=20):
    """
    간단한 3D 장면 데이터 생성 (실제 데이터셋 없이 테스트용)
    중앙에 구 형태의 객체를 배치하고 원형 카메라 경로 생성
    """
    data = []
    H, W = 200, 200
    focal = 200.0
    
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        radius = 4.0
        
        # Camera-to-world 변환 행렬 (y축 위로, 원형 경로)
        cam_pos = np.array([
            radius * np.cos(angle),
            0.5,
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
        
        # 간단한 타겟 이미지 생성 (중앙에 밝은 영역)
        y, x = np.ogrid[-1:1:H*1j, -1:1:H*1j]
        img = np.zeros((H, W, 3), dtype=np.float32)
        
        # 구 형태 렌더링 (간단한 램버트 쉐이딩)
        r = np.sqrt(x**2 + y**2)
        mask = r < 0.5
        img[mask] = 0.8
        
        data.append({
            'image': torch.from_numpy(img).to(device),
            'c2w': torch.from_numpy(c2w).to(device),
            'focal': focal,
            'H': H,
            'W': W
        })
    
    return data

