import sys
import os
import functools
import math
import warnings
from collections import OrderedDict, Iterable, Mapping
from itertools import islice
import operator
import torch
import torch.nn as nn
from seg_opr.conv_2_5d import Malleable_Conv2_5D_Depth
from torch.nn import functional as F


import time
from engine.logger import get_logger

logger = get_logger()

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None,
                 bn_eps=1e-5, bn_momentum=0.1, downsample=None, inplace=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual

        out = self.relu_inplace(out)

        return out


class DSequential(nn.Module):
    def __init__(self, *args):
        super(DSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return DSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(DSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input, depth, camera_params, coord):
        flag = True
        for module in self._modules.values():
            if flag:
                input = module(input, depth, camera_params, coord)
                flag = False
            else:
                input = module(input)
        return input

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)

        return out
class DBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True, pixel_size=1):
        super(DBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = Malleable_Conv2_5D_Depth(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, pixel_size=pixel_size, anchor_init=[-2.,-1.,0.,1.,2], fix_center=True)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x, depth, camera_params, coord):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, depth, camera_params)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, first_block, layers, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()
        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        self.bn1 = norm_layer(stem_width * 2 if deep_stem else 64, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, first_block, norm_layer, 64, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum, pixel_size=4)
        self.layer2 = self._make_layer(block, first_block, norm_layer, 128, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum, pixel_size=8)
        self.layer3 = self._make_layer(block, first_block, norm_layer, 256, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum, pixel_size=16)
        self.layer4 = self._make_layer(block, first_block, norm_layer, 512, layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum, pixel_size=16)

    def _make_layer(self, block, first_block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1,pixel_size=4):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
            )

        layers = []
        layers.append(first_block(self.inplanes, planes, stride, norm_layer, bn_eps,
                            bn_momentum, downsample, inplace, pixel_size=pixel_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, bn_eps=bn_eps,
                                bn_momentum=bn_momentum, inplace=inplace))

        return DSequential(*layers)

    def forward(self, x, depth=None, camera_params=None, coord=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        blocks = []
        current_depth = F.interpolate(depth, scale_factor=1 / 4, mode='nearest')
        current_coord = F.interpolate(coord, scale_factor=1 / 4, mode='nearest')
        x = self.layer1(x, current_depth, camera_params, current_coord);
        blocks.append(x)
        current_depth = F.interpolate(depth, scale_factor=1 / 4, mode='nearest')
        current_coord = F.interpolate(coord, scale_factor=1 / 4, mode='nearest')
        #print(current_depth.shape)
        x = self.layer2(x, current_depth, camera_params, current_coord);
        blocks.append(x)
        current_depth = F.interpolate(depth, scale_factor=1 / 8, mode='nearest')
        current_coord = F.interpolate(coord, scale_factor=1 / 8, mode='nearest')
        x = self.layer3(x, current_depth, camera_params, current_coord);
        blocks.append(x)
        current_depth = F.interpolate(depth, scale_factor=1 / 16, mode='nearest')
        current_coord = F.interpolate(coord, scale_factor=1 / 16, mode='nearest')
        x = self.layer4(x, current_depth, camera_params, current_coord);
        blocks.append(x)

        return blocks

def load_model(model, model_file, is_restore=False):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file)


        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    state_dict = {}
    for k, v in raw_state_dict.items():
        state_dict[k.replace('.bn.', '.')] = v
        if k.find('.0.conv2.') >= 0:
            state_dict[k.replace('weight', 'weight_0')] = v
            state_dict[k.replace('weight', 'weight_1')] = v
            state_dict[k.replace('weight', 'weight_2')] = v
    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    # print('ckpt_keys', own_keys)
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logger.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys if k.find('num_batches_tracked')<0)))

    if len(unexpected_keys) > 0:
        logger.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys if k.find('num_batches_tracked')<0)))

    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model
def resnet18(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet34(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet50(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, DBottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet101(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck,DBottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet152(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck,DBottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model
