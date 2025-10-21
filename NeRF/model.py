import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    """
    Neural Radiance Field 모델
    위치(x, y, z)와 방향(theta, phi)을 입력으로 받아
    색상(RGB)과 밀도(sigma)를 출력
    """
    def __init__(
        self,
        pos_dim=3,
        view_dim=3,
        pos_L=10,
        view_L=4,
        hidden_dim=256,
        use_viewdirs=True
    ):
        super(NeRF, self).__init__()
        
        self.pos_L = pos_L
        self.view_L = view_L
        self.use_viewdirs = use_viewdirs
        
        # Positional encoding 차원
        self.pos_input_dim = pos_dim * (2 * pos_L + 1)
        self.view_input_dim = view_dim * (2 * view_L + 1) if use_viewdirs else 0
        
        # 위치 인코딩 네트워크 (8개 레이어)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.pos_input_dim, hidden_dim)] +
            [nn.Linear(hidden_dim, hidden_dim) if i not in [4] else 
             nn.Linear(hidden_dim + self.pos_input_dim, hidden_dim) 
             for i in range(7)]
        )
        
        # 밀도 출력 레이어
        self.sigma_linear = nn.Linear(hidden_dim, 1)
        
        # 방향 의존 색상 네트워크
        if use_viewdirs:
            self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
            self.view_linear = nn.Linear(hidden_dim + self.view_input_dim, hidden_dim // 2)
            self.rgb_linear = nn.Linear(hidden_dim // 2, 3)
        else:
            self.rgb_linear = nn.Linear(hidden_dim, 3)
    
    def positional_encoding(self, x, L):
        """위치 인코딩: gamma(p) = [sin(2^0*pi*p), cos(2^0*pi*p), ..., sin(2^(L-1)*pi*p), cos(2^(L-1)*pi*p)]"""
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * torch.pi * x))
            out.append(torch.cos(2 ** j * torch.pi * x))
        return torch.cat(out, dim=-1)
    
    def forward(self, x, d=None):
        """
        Args:
            x: (B, 3) 위치 좌표
            d: (B, 3) 방향 벡터 (정규화됨)
        Returns:
            rgb: (B, 3) RGB 색상
            sigma: (B, 1) 밀도
        """
        # 위치 인코딩
        input_pts = self.positional_encoding(x, self.pos_L)
        h = input_pts
        
        # 위치 처리 네트워크
        for i, layer in enumerate(self.pts_linears):
            h = layer(h)
            h = F.relu(h)
            if i == 4:
                # Skip connection
                h = torch.cat([input_pts, h], dim=-1)
        
        # 밀도 출력
        sigma = self.sigma_linear(h)
        sigma = F.relu(sigma)
        
        # RGB 출력
        if self.use_viewdirs and d is not None:
            feature = self.feature_linear(h)
            input_views = self.positional_encoding(d, self.view_L)
            h = torch.cat([feature, input_views], dim=-1)
            h = self.view_linear(h)
            h = F.relu(h)
            rgb = self.rgb_linear(h)
        else:
            rgb = self.rgb_linear(h)
        
        rgb = torch.sigmoid(rgb)
        
        return rgb, sigma


def get_rays(H, W, focal, c2w):
    """
    카메라 ray 생성
    Args:
        H, W: 이미지 높이, 너비
        focal: 초점 거리
        c2w: (4, 4) camera-to-world 변환 행렬
    Returns:
        rays_o: (H, W, 3) ray origins
        rays_d: (H, W, 3) ray directions
    """
    # c2w와 같은 device 사용
    device = c2w.device
    
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='xy'
    )
    
    # 카메라 좌표계에서의 방향 벡터
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    
    # 월드 좌표계로 변환
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return rays_o, rays_d


def render_rays(nerf_model, rays_o, rays_d, near, far, N_samples=64, device='cpu'):
    """
    Ray를 따라 볼륨 렌더링 수행
    Args:
        nerf_model: NeRF 모델
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        near, far: near/far plane 거리
        N_samples: ray당 샘플 수
    Returns:
        rgb_map: (N_rays, 3) 렌더링된 RGB 색상
        depth_map: (N_rays,) 깊이 맵
    """
    N_rays = rays_o.shape[0]
    
    # Ray를 따라 샘플링할 t 값들 생성
    t_vals = torch.linspace(0., 1., steps=N_samples, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand([N_rays, N_samples])
    
    # Stratified sampling
    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
    lower = torch.cat([z_vals[..., :1], mids], dim=-1)
    t_rand = torch.rand(z_vals.shape, device=device)
    z_vals = lower + (upper - lower) * t_rand
    
    # 3D 포인트 계산
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # (N_rays, N_samples, 3)
    
    # 방향 벡터 정규화
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, :].expand(pts.shape)
    
    # NeRF 모델 실행
    pts_flat = pts.reshape(-1, 3)
    viewdirs_flat = viewdirs.reshape(-1, 3)
    
    rgb, sigma = nerf_model(pts_flat, viewdirs_flat)
    rgb = rgb.reshape(N_rays, N_samples, 3)
    sigma = sigma.reshape(N_rays, N_samples)
    
    # 볼륨 렌더링
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full((N_rays, 1), 1e10, device=device)], dim=-1)
    
    alpha = 1.0 - torch.exp(-sigma * dists)
    
    # T_i = exp(-sum(sigma_j * delta_j)) for j < i
    transmittance = torch.cumprod(
        torch.cat([torch.ones((N_rays, 1), device=device), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]
    
    weights = alpha * transmittance
    
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)
    
    # 배경 흰색 처리
    rgb_map = rgb_map + (1.0 - acc_map[..., None])
    
    return rgb_map, depth_map

