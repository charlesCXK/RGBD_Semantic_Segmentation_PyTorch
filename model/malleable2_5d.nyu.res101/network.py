# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict
from config import config
from base_model import Bottleneck
from resnet import resnet101
from seg_opr.geo_utils import Plane2Space
from seg_opr.conv_2_5d import Malleable_Conv2_5D_Depth

class DeepLab(nn.Module):
    def __init__(self, out_planes, criterion, norm_layer, pretrained_model=None):
        super(DeepLab, self).__init__()
        self.backbone = resnet101(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=True, stem_width=64)
        self.dilate = 2

        #self.backbone.layer4.apply(partial(self._nostride_dilate, dilate= self.dilate ) )
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.seg_layer = nn.Conv2d(128, out_planes, kernel_size=1, stride=1, padding=0)
        self.seg_layer_2 = nn.Conv2d(512, out_planes, kernel_size=1, stride=1, padding=0)
        self.upsample_1 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 3, 1, 1),
                                        norm_layer(512),
                                        nn.ReLU(inplace=True))

        self.upsample_2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                        norm_layer(256),
                                        nn.ReLU(inplace=True))

        self.upsample_3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                        norm_layer(128),
                                        nn.ReLU(inplace=True))
        self.embed = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1),
                                        norm_layer(512),
                                        nn.ReLU(inplace=True))
        self.sup_stride = 4
        self.node_num = (config.image_height // (8 * self.sup_stride)) * (config.image_width // (8 * self.sup_stride))
        self.aspp = ASPP(512, 512, [6, 12, 18], norm_act=norm_layer)
        self.merge_module = Merge_Module([256, 512, 1024], 512, 512, norm_layer)
        self.task_module = Task_Module(512, 512, norm_layer)
        self.task_module.layer_1.apply(partial(self._nostride_dilate, dilate= self.dilate ))
        self.task_module.layer_2.apply(partial(self._nostride_dilate, dilate= self.dilate ))
        self.business_layer = []
        self.business_layer.append(self.seg_layer)
        self.business_layer.append(self.seg_layer_2)
        self.business_layer.append(self.merge_module)
        self.business_layer.append(self.task_module)
        self.business_layer.append(self.upsample_1)
        self.business_layer.append(self.upsample_2)
        self.business_layer.append(self.upsample_3)
        self.business_layer.append(self.embed)
        self.business_layer.append(self.aspp)
        self.criterion = criterion

    def forward(self, data, depth, coordinate, camera_params, label=None):
        blocks = self.backbone(data, depth, camera_params, coordinate)
        feat = blocks[-1]
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear')
        feat_8x = self.upsample_1(feat)
        feat = self.merge_module(blocks, feat_8x)
        feat = self.task_module(feat)
        feat_8x_2 = self.embed(feat)
        depth_8x = F.interpolate(depth, scale_factor=1/8, mode='bilinear')
        coord_8x = F.interpolate(coordinate, scale_factor = 1/8, mode = 'nearest')
        feat = self.aspp(feat_8x_2)

        feat = F.interpolate(feat, scale_factor=2, mode='bilinear')
        feat = self.upsample_2(feat)
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear')
        feat = self.upsample_3(feat)
        aspp_fm = self.seg_layer(feat)
        aspp_fm = F.interpolate(aspp_fm, scale_factor=2, mode='bilinear',
                                align_corners=True)

        aux_fm = self.seg_layer_2(feat_8x_2)
        aux_fm = F.interpolate(aux_fm, scale_factor=8, mode='bilinear')

        if label is not None:
            loss = self.criterion(aspp_fm, label)
            loss_2 = self.criterion(aux_fm, label)
            return loss+0.4*loss_2

        return aspp_fm

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d) or isinstance(m, Malleable_Conv2_5D_Depth):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


class Merge_Module(nn.Module):
    def __init__(self, in_channels, feat_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(Merge_Module, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(feat_channels + 3 * out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, blocks, feat):
        x1, x2 , x3 = blocks[0], blocks[1], blocks[2]
        h, w = feat.size()[2], feat.size(3)
        x1 = F.interpolate(x1, size=(h,w), mode='bilinear')
        x2 = F.interpolate(x2, size=(h,w), mode='bilinear')
        x3 = F.interpolate(x3, size=(h,w), mode='bilinear')
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        out = torch.cat((x1, x2, x3, feat), dim=1)
        out = self.conv4(out)
        return out

class Task_Module(nn.Module):
    def __init__(self, in_channels, embed_channels, norm_layer):
        super(Task_Module, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4,
                      kernel_size=1, stride=1, bias=False),
            norm_layer(in_channels * 4)
        )
        self.layer_1 = Bottleneck(in_channels, embed_channels, norm_layer=norm_layer, downsample=self.downsample)
        self.layer_2 = Bottleneck(in_channels * 4, embed_channels, norm_layer=norm_layer)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x

class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size
        
        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)
        
        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)
        
        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)
    
    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = F.relu(out)
        out = self.red_conv(out)
        
        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))
        
        out += pool
        out = self.red_bn(out)
        out = F.relu(out)
        return out
    
    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )
            
            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool
