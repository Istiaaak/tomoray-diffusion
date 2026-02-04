import torch
import torch.nn as nn
from .unet import Unet2d
import torch.nn.functional as F
import numpy as np

class Fusion(nn.Module):
    def __init__(self, volume_shape=(64, 64, 64), vol_size_mm=(240, 240, 240), det_size_mm=240.0, n_features=128):
        super().__init__()
        
        self.volume_shape = volume_shape
        self.vol_size_mm = vol_size_mm
        self.det_size_mm = det_size_mm

        # CT have only one channel
        self.unet = Unet2d(n_channels=1, n_classes=n_features)

    
    def get_sampling_grid(self, angles, device):

        batch_size, n_views = angles.shape
        D, H, W = self.volume_shape

        d_range = torch.linspace(-self.vol_size_mm[0]/2, self.vol_size_mm[0]/2, D, device=device)
        h_range = torch.linspace(-self.vol_size_mm[1]/2, self.vol_size_mm[1]/2, H, device=device)
        w_range = torch.linspace(-self.vol_size_mm[2]/2, self.vol_size_mm[2]/2, W, device=device)

        grid_d, grid_h, grid_w = torch.meshgrid(d_range, h_range, w_range, indexing="ij")
        # Width (X), Height (Y), Depth (Z)
        points_3d = torch.stack([grid_w.flatten(), grid_h.flatten(), grid_d.flatten()], dim=0)

        all_grids = []

        for b in range(batch_size):
            view_grids = []
            for n in range(n_views):
                angle = angles[b, n]


                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)

                R = torch.tensor([
                    [cos_a, -sin_a, 0],
                    [sin_a,  cos_a, 0],
                    [0,      0,     1]
                ], device=device, dtype=torch.float32)

                rot_points = torch.matmul(R, points_3d)


                u = rot_points[0, :]
                v = rot_points[2, :]

                u_norm = 2 * (u / self.det_size_mm)
                v_norm = 2 * (v / self.det_size_mm)


                grid = torch.stack([u_norm, v_norm], dim=-1)

                grid = grid.view(D, H, W , 2)
                
                view_grids.append(grid)

            all_grids.append(torch.stack(view_grids, dim=0))
        
        return torch.stack(all_grids, dim=0)
    

    def project_and_fuse(self, features, angles):
        
        batch_size, n_views, C, h, w = features.shape
        D, H, W = self.volume_shape

        grids = self.get_sampling_grid(angles, features.device)

        features_flat = features.view(batch_size*n_views, C, h, w)

        grids_flat = grids.view(batch_size*n_views, 1,  D*H*W, 2)

        sampled = F.grid_sample(features_flat, grids_flat, align_corners=True, padding_mode='zeros')
        
        sampled_vol = sampled.view(batch_size, n_views, C, D, H, W)
        fused = torch.mean(sampled_vol, dim=1)
        # (B, C, D, H, W) -> (B, C, H, W, D)
        fused_aligned = fused.permute(0, 1, 3, 4, 2)
        return fused


    def forward(self, x_rays, angles):

        batch_size, n_views, C, H, W = x_rays.shape

        x_reshaped = x_rays.view(batch_size * n_views, C, H, W)

        features_2d = self.unet(x_reshaped)

        n_feat = features_2d.shape[1]
        features_2d = features_2d.view(batch_size, n_views, n_feat, H, W)

        return self.project_and_fuse(features_2d, angles)

    def forward_bypass(self, x_rays, angles):

        return self.project_and_fuse(x_rays, angles)



def get_gt_aligned_xyz(data: dict) -> np.ndarray:
    img_xyz = data["image"].astype(np.float32)
    preprocess = data.get("preprocess", None)
    if preprocess is not None and preprocess.get("image_matches_projections", False):
        return img_xyz
    if preprocess is None: return img_xyz
    ori = preprocess.get("internal_orientation") or preprocess.get("orientation_used")
    if ori == "LPI": img_xyz = img_xyz[::-1, ::-1, ::-1].copy()
    flip_used = preprocess.get("flip_used", "none")
    if flip_used == "x": img_xyz = img_xyz[::-1, :, :].copy()
    elif flip_used == "y": img_xyz = img_xyz[:, ::-1, :].copy()
    elif flip_used == "z": img_xyz = img_xyz[:, :, ::-1].copy()
    return img_xyz

def show_yz(vol_xyz, x, ax, title):
    ax.imshow(vol_xyz[x, :, :].T, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")