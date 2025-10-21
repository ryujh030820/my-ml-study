"""
간단한 샘플 데이터셋 생성 스크립트
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw


def create_sample_dataset(data_dir='./data'):
    """
    실제 이미지가 있는 작은 샘플 데이터셋 생성
    """
    print("=" * 60)
    print("샘플 데이터셋 생성 중...")
    print("=" * 60)
    
    lego_dir = os.path.join(data_dir, 'lego')
    os.makedirs(lego_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        n_views = 20 if split == 'train' else 5
        frames = []
        
        print(f"\n{split} 데이터 생성 중... ({n_views} 뷰)")
        
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
            
            # 간단한 이미지 렌더링 (중앙에 3D 구조)
            img_size = 400
            img = Image.new('RGBA', (img_size, img_size), (255, 255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # 구체를 투영 (간단한 원 그리기)
            center_x = img_size // 2
            center_y = img_size // 2
            
            # 각도에 따른 간단한 shading (view-dependent)
            brightness = int(128 + 127 * np.cos(angle))
            side_brightness = int(128 + 127 * np.sin(angle))
            
            # 여러 레이어로 3D 구조 표현
            # 뒤쪽 큰 구체
            draw.ellipse([center_x-80, center_y-80, center_x+80, center_y+80], 
                        fill=(brightness, 100, 150, 255))
            
            # 중간 구체
            draw.ellipse([center_x-50, center_y-50, center_x+50, center_y+50], 
                        fill=(100, brightness, 180, 255))
            
            # 작은 구체들 (각도에 따라 위치 변경)
            offset_x = int(40 * np.cos(angle))
            offset_y = int(40 * np.sin(angle))
            draw.ellipse([center_x-25+offset_x, center_y-25+offset_y, 
                         center_x+25+offset_x, center_y+25+offset_y], 
                        fill=(side_brightness, 180, brightness, 255))
            
            # 상단 작은 구체
            draw.ellipse([center_x-20, center_y-90, center_x+20, center_y-50], 
                        fill=(180, side_brightness, 100, 255))
            
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
        
        print(f"  ✓ {split}: {n_views} 이미지 + {os.path.basename(json_path)} 생성")
    
    print("\n" + "=" * 60)
    print(f"✅ 샘플 데이터셋이 {lego_dir} 에 생성되었습니다!")
    print("=" * 60)
    print(f"총 이미지 수: {20 + 5 + 5} 장")
    print("\n이제 다음 명령어로 학습을 시작할 수 있습니다:")
    print("  python train.py")
    print("=" * 60)


if __name__ == '__main__':
    create_sample_dataset()

