import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def _ntuple(n):
    def parse(x):
        if isinstance(x, list) or isinstance(x, tuple):
            return x
        return tuple([x]*n)
    return parse
_pair = _ntuple(2)

class Conv2_5D_Depth(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, pixel_size=1):
        super(Conv2_5D_Depth, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0]*self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        assert self.kernel_size_prod%2==1

        self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, depth, camera_params):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)//self.stride[0]+1
        out_W = (W+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)//self.stride[1]+1

        intrinsic = camera_params['intrinsic']
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride) # N*(C*kh*kw)*(out_H*out_W)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H*out_W)
        depth_col = F.unfold(depth, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride) # N*(kh*kw)*(out_H*out_W)
        valid_mask = 1-depth_col.eq(0.).to(torch.float32)

        valid_mask = valid_mask*valid_mask[:, self.kernel_size_prod//2, :].view(N,1,out_H*out_W)
        depth_col *= valid_mask
        valid_mask = valid_mask.view(N,1,self.kernel_size_prod,out_H*out_W)

        center_depth = depth_col[:,self.kernel_size_prod//2,:].view(N,1,out_H*out_W)
        # grid_range = self.pixel_size * center_depth / (intrinsic['fx'].view(N,1,1) * camera_params['scale'].view(N,1,1))
        grid_range = self.pixel_size * self.dilation[0] * center_depth / intrinsic['fx'].view(N,1,1)
        
        mask_0 = torch.abs(depth_col - (center_depth + grid_range)).le(grid_range/2).view(N,1,self.kernel_size_prod,out_H*out_W).to(torch.float32)
        mask_1 = torch.abs(depth_col - (center_depth             )).le(grid_range/2).view(N,1,self.kernel_size_prod,out_H*out_W).to(torch.float32)
        mask_1 = (mask_1 + 1- valid_mask).clamp(min=0., max=1.)
        mask_2 = torch.abs(depth_col - (center_depth - grid_range)).le(grid_range/2).view(N,1,self.kernel_size_prod,out_H*out_W).to(torch.float32)
        output  = torch.matmul(self.weight_0.view(-1,C*self.kernel_size_prod), (x_col*mask_0).view(N, C*self.kernel_size_prod, out_H*out_W))
        output += torch.matmul(self.weight_1.view(-1,C*self.kernel_size_prod), (x_col*mask_1).view(N, C*self.kernel_size_prod, out_H*out_W))
        output += torch.matmul(self.weight_2.view(-1,C*self.kernel_size_prod), (x_col*mask_2).view(N, C*self.kernel_size_prod, out_H*out_W))
        output = output.view(N,-1,out_H,out_W)
        if self.bias:
            output += self.bias.view(1,-1,1,1)
        return output

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Malleable_Conv2_5D_Depth(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, pixel_size=1, anchor_init=[-2.,-1.,0.,1.,2.], scale_const=100, fix_center=False, adjust_to_scale=False):
        super(Malleable_Conv2_5D_Depth, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = self.kernel_size[0]*self.kernel_size[1]
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pixel_size = pixel_size
        self.fix_center = fix_center
        self.adjust_to_scale = adjust_to_scale
        assert self.kernel_size_prod%2==1

        self.weight_0 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_1 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.weight_2 = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
        self.depth_anchor = Parameter(torch.tensor(anchor_init, requires_grad=True).view(1,5,1,1))
        # self.depth_bias = Parameter(torch.tensor([0.,0.,0.,0.,0.], requires_grad=True).view(1,5,1,1))
        self.temperature = Parameter(torch.tensor([1.], requires_grad=True))
        self.kernel_weight = Parameter(torch.tensor([0.,0.,0.], requires_grad=True))
        self.scale_const = scale_const
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, depth, camera_params):
        N, C, H, W = x.size(0), x.size(1), x.size(2), x.size(3)
        out_H = (H+2*self.padding[0]-self.dilation[0]*(self.kernel_size[0]-1)-1)//self.stride[0]+1
        out_W = (W+2*self.padding[1]-self.dilation[1]*(self.kernel_size[1]-1)-1)//self.stride[1]+1

        intrinsic = camera_params['intrinsic']
        x_col = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride) # N*(C*kh*kw)*(out_H*out_W)
        x_col = x_col.view(N, C, self.kernel_size_prod, out_H*out_W)
        depth_col = F.unfold(depth, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride) # N*(kh*kw)*(out_H*out_W)
        valid_mask = 1-depth_col.eq(0.).to(torch.float32)

        valid_mask = valid_mask*valid_mask[:, self.kernel_size_prod//2, :].view(N,1,out_H*out_W)
        depth_col *= valid_mask
        valid_mask = valid_mask.view(N,1,self.kernel_size_prod,out_H*out_W)

        center_depth = depth_col[:,self.kernel_size_prod//2,:].view(N,1,out_H*out_W)
        if self.adjust_to_scale:
            grid_range = self.pixel_size * self.dilation[0] * center_depth / (intrinsic['fx'].view(N,1,1) * camera_params['scale'].view(N,1,1))
        else:
            grid_range = self.pixel_size * self.dilation[0] * center_depth / intrinsic['fx'].view(N,1,1)
        depth_diff = (depth_col - center_depth).view(N, 1, self.kernel_size_prod, out_H*out_W) # N*1*(kh*kw)*(out_H*out_W)
        relative_diff = depth_diff*self.scale_const/(1e-5 + grid_range.view(N,1,1,out_H*out_W)*self.scale_const)
        depth_logit = -( ((relative_diff - self.depth_anchor).pow(2)) / (1e-5 + torch.clamp(self.temperature, min=0.)) ) # N*5*(kh*kw)*(out_H*out_W)
        if self.fix_center:
            depth_logit[:,2,:,:] = -( ((relative_diff - 0.).pow(2)) / (1e-5 + torch.clamp(self.temperature, min=0.)) ).view(N,self.kernel_size_prod,out_H*out_W)

        depth_out_range_0 = (depth_diff<self.depth_anchor[0,0,0,0]).to(torch.float32).view(N,self.kernel_size_prod,out_H*out_W)
        depth_out_range_4 = (depth_diff>self.depth_anchor[0,4,0,0]).to(torch.float32).view(N,self.kernel_size_prod,out_H*out_W)
        depth_logit[:,0,:,:] = depth_logit[:,0,:,:]*(1 - 2*depth_out_range_0)
        depth_logit[:,4,:,:] = depth_logit[:,4,:,:]*(1 - 2*depth_out_range_4)

        depth_class = F.softmax(depth_logit, dim=1) # N*5*(kh*kw)*(out_H*out_W)
        
        mask_0 = depth_class[:,1,:,:].view(N,1,self.kernel_size_prod,out_H*out_W).to(torch.float32)
        mask_1 = depth_class[:,2,:,:].view(N,1,self.kernel_size_prod,out_H*out_W).to(torch.float32)
        mask_2 = depth_class[:,3,:,:].view(N,1,self.kernel_size_prod,out_H*out_W).to(torch.float32)

        invalid_mask_bool = valid_mask.eq(0.)

        mask_0 = mask_0*valid_mask
        mask_1 = mask_1*valid_mask
        mask_2 = mask_2*valid_mask
        mask_0[invalid_mask_bool] = 1./5.
        mask_1[invalid_mask_bool] = 1./5.
        mask_2[invalid_mask_bool] = 1./5.

        weight = F.softmax(self.kernel_weight, dim=0) * 3 #???
        output  = torch.matmul(self.weight_0.view(-1,C*self.kernel_size_prod), (x_col*mask_0).view(N, C*self.kernel_size_prod, out_H*out_W)) * weight[0]
        output += torch.matmul(self.weight_1.view(-1,C*self.kernel_size_prod), (x_col*mask_1).view(N, C*self.kernel_size_prod, out_H*out_W)) * weight[1]
        output += torch.matmul(self.weight_2.view(-1,C*self.kernel_size_prod), (x_col*mask_2).view(N, C*self.kernel_size_prod, out_H*out_W)) * weight[2]
        output = output.view(N,-1,out_H,out_W)
        if self.bias:
            output += self.bias.view(1,-1,1,1)
        return output

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
