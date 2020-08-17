import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Plane2Space(nn.Module):
    def __init__(self):
        super(Plane2Space, self).__init__()

    def forward(self, depth, coordinate, camera_params):
        valid_mask = 1-depth.eq(0.).to(torch.float32)

        depth = torch.clamp(depth, min=1e-5)
        N, H, W = depth.size(0), depth.size(2), depth.size(3)
        intrinsic = camera_params['intrinsic']

        K_inverse = depth.new_zeros(N, 3, 3)
        K_inverse[:,0,0] = 1./intrinsic['fx']
        K_inverse[:,1,1] = 1./intrinsic['fy']
        K_inverse[:,2,2] = 1.
        if 'cx' in intrinsic:
            K_inverse[:,0,2] = -intrinsic['cx']/intrinsic['fx']
            K_inverse[:,1,2] = -intrinsic['cy']/intrinsic['fy']
        elif 'u0' in intrinsic:
            K_inverse[:,0,2] = -intrinsic['u0']/intrinsic['fx']
            K_inverse[:,1,2] = -intrinsic['v0']/intrinsic['fy']
        coord_3d = torch.matmul(K_inverse, (coordinate.float()*depth.float()).view(N,3,H*W)).view(N,3,H,W).contiguous()
        coord_3d *= valid_mask

        return coord_3d

class Disp2Depth(nn.Module):
    def __init__(self, min_disp=0.01, max_disp=256):
        self.min_disp = min_disp
        self.max_disp = max_disp
        super(Disp2Depth, self).__init__()

    def forward(self, disp, camera_params):
        N = disp.size(0)
        intrinsic, extrinsic = camera_params['intrinsic'], camera_params['extrinsic']
        valid_mask = 1 - disp.eq(0.).to(torch.float32)
        depth = (extrinsic['baseline'] * intrinsic['fx']).view(N, 1, 1, 1).cuda() / torch.clamp(disp, self.min_disp, self.max_disp)
        depth *= valid_mask
        return depth