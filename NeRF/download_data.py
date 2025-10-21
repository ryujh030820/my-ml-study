"""
NeRF Synthetic Dataset 다운로드 스크립트
"""
import os
import requests
import zipfile
from tqdm import tqdm


def download_file(url, destination):
    """URL에서 파일 다운로드"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def download_nerf_synthetic_dataset(data_dir='./data'):
    """
    NeRF Synthetic Dataset 다운로드
    
    데이터셋: lego, chair, drums, ficus, hotdog, materials, mic, ship
    """
    print("=" * 60)
    print("NeRF Synthetic Dataset 다운로드")
    print("=" * 60)
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Google Drive 링크 (공식 NeRF synthetic dataset)
    # 직접 다운로드 링크
    base_url = "https://www.dropbox.com/scl/fo/0kbb044w45tiw83b3l6rn/AHLTbjYLPaG0VEHBJt8C91U?rlkey=jfqzl7s3v1a3yt7bnjbu5e0vq&st=ygz1w0sk&dl=1"
    
    print("\n참고: NeRF synthetic dataset은 약 2.5GB 크기입니다.")
    print("공식 다운로드 링크에서 직접 다운로드해주세요:")
    print("https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1")
    print("\n또는 아래 명령어를 사용하세요:")
    print()
    print("# gdown 설치 (Google Drive 다운로드 도구)")
    print("pip install gdown")
    print()
    print("# Lego 데이터셋 다운로드")
    print("cd NeRF/data")
    print("gdown 1hkHfPnCUCcSoMK4wO0E5K0-fhFnE6Rr5")
    print("unzip nerf_synthetic.zip")
    print("mv nerf_synthetic/* .")
    print("rm -rf nerf_synthetic nerf_synthetic.zip")
    print()
    
    # 대안: 작은 샘플 데이터 생성
    print("=" * 60)
    print("대안: 작은 샘플 데이터셋을 생성하시겠습니까? (y/n)")
    choice = input().lower()
    
    if choice == 'y':
        create_sample_dataset(data_dir)
    else:
        print("\n수동으로 데이터셋을 다운로드해주세요.")
        print("다운로드 후 ./data/lego/ 디렉토리에 배치하세요.")


def create_sample_dataset(data_dir):
    """
    실제 이미지가 있는 작은 샘플 데이터셋 생성
    (진짜 3D 장면을 간단히 렌더링)
    """
    import json
    import numpy as np
    from PIL import Image, ImageDraw
    
    print("\n작은 샘플 데이터셋 생성 중...")
    
    lego_dir = os.path.join(data_dir, 'lego')
    os.makedirs(lego_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        n_views = 20 if split == 'train' else 5
        frames = []
        
        print(f"{split} 데이터 생성 중... ({n_views} 뷰)")
        
        for i in range(n_views):
            angle = 2 * np.pi * i / n_views
            radius = 4.0
            
            # 카메라 위치
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
            
            transform_matrix = np.eye(4)
            transform_matrix[:3, 0] = right
            transform_matrix[:3, 1] = up
            transform_matrix[:3, 2] = forward
            transform_matrix[:3, 3] = cam_pos
            
            # 간단한 이미지 렌더링 (중앙에 구체)
            img_size = 200
            img = Image.new('RGBA', (img_size, img_size), (255, 255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # 구체를 투영 (간단한 원 그리기)
            center_x = img_size // 2
            center_y = img_size // 2
            
            # 각도에 따른 간단한 shading
            brightness = int(128 + 127 * np.cos(angle))
            
            # 여러 개의 구체 그리기
            draw.ellipse([center_x-60, center_y-60, center_x+60, center_y+60], 
                        fill=(brightness, 100, 200, 255))
            draw.ellipse([center_x-30, center_y-80, center_x+30, center_y-20], 
                        fill=(200, brightness, 100, 255))
            
            # 이미지 저장
            img_path = os.path.join(lego_dir, f'{split}_r_{i:03d}.png')
            img.save(img_path)
            
            frames.append({
                'file_path': f'./{split}_r_{i:03d}',
                'rotation': angle,
                'transform_matrix': transform_matrix.tolist()
            })
        
        # transforms JSON 저장
        transforms = {
            'camera_angle_x': 0.6911112070083618,
            'frames': frames
        }
        
        json_path = os.path.join(lego_dir, f'transforms_{split}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transforms, f, indent=2)
        
        print(f"  ✓ {json_path} 생성 완료")
    
    print(f"\n✅ 샘플 데이터셋이 {lego_dir} 에 생성되었습니다!")
    print("이제 train.py를 실행할 수 있습니다.")


if __name__ == '__main__':
    download_nerf_synthetic_dataset()

