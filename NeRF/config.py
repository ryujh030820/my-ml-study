"""
NeRF 학습 및 테스트 설정
"""

# 데이터셋 설정
DATASET_CONFIG = {
    'lego': {
        'datadir': './nerf_synthetic/lego',
        'near': 2.0,
        'far': 6.0,
        'render_radius': 4.0,
    },
    'chair': {
        'datadir': './nerf_synthetic/chair',
        'near': 2.0,
        'far': 6.0,
        'render_radius': 3.0,
    },
    'drums': {
        'datadir': './nerf_synthetic/drums',
        'near': 2.0,
        'far': 6.0,
        'render_radius': 3.5,
    },
    'ficus': {
        'datadir': './nerf_synthetic/ficus',
        'near': 2.0,
        'far': 6.0,
        'render_radius': 3.0,
    },
    'hotdog': {
        'datadir': './nerf_synthetic/hotdog',
        'near': 2.0,
        'far': 6.0,
        'render_radius': 3.0,
    },
    'materials': {
        'datadir': './nerf_synthetic/materials',
        'near': 2.0,
        'far': 6.0,
        'render_radius': 3.0,
    },
    'mic': {
        'datadir': './nerf_synthetic/mic',
        'near': 2.0,
        'far': 6.0,
        'render_radius': 3.5,
    },
    'ship': {
        'datadir': './nerf_synthetic/ship',
        'near': 2.0,
        'far': 6.0,
        'render_radius': 3.0,
    },
}

# 모델 설정
MODEL_CONFIG = {
    'pos_L': 10,
    'view_L': 4,
    'hidden_dim': 256,
    'use_viewdirs': True,
}

# 학습 설정
TRAIN_CONFIG = {
    'n_epochs': 30,
    'batch_size': 1024,
    'lr': 5e-4,
    'N_samples': 64,
    'save_dir': './checkpoints',
    'log_interval': 5,
}

# 렌더링 설정
RENDER_CONFIG = {
    'N_samples': 128,  # 테스트 시 더 많은 샘플 사용
    'chunk': 1024,
    'n_video_frames': 30,
}

