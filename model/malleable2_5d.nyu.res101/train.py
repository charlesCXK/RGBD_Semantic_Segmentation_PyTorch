from __future__ import division
import os.path as osp
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
# from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from config import config
from dataloader import get_train_loader
from network import DeepLab
from nyu import NYUv2
from utils.init_func import init_weight, group_weight
from engine.lr_policy import PolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

parser = argparse.ArgumentParser()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if dist.get_rank()==0:
        writer = SummaryWriter(config.log_dir)

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, NYUv2)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

    if len(engine.devices)==1:
        BatchNorm2d = nn.BatchNorm2d
    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    model = DeepLab(config.num_classes, criterion=criterion,
                pretrained_model=config.pretrained_model,
                norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr# * engine.world_size

    params_list = []
    params_list = group_weight(params_list, model.backbone,
                               BatchNorm2d, base_lr)
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d, base_lr)

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    optimizer.param_groups[2]['lr'] = base_lr#*2

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

    if engine.distributed:
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    model.train()
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(engine.state.epoch, config.nepochs):
        if dist.get_rank()==0:
            print('layer1', 
                'anchor', model.module.backbone.layer1[0].conv2.depth_anchor.view(5).tolist(), 
                'temperature', model.module.backbone.layer1[0].conv2.temperature.view(1).tolist(),
                'weight', F.softmax(model.module.backbone.layer1[0].conv2.kernel_weight).view(3).tolist(),
                )
            print('layer2', 
                'anchor', model.module.backbone.layer2[0].conv2.depth_anchor.view(5).tolist(), 
                'temperature', model.module.backbone.layer2[0].conv2.temperature.view(1).tolist(),
                'weight', F.softmax(model.module.backbone.layer2[0].conv2.kernel_weight).view(3).tolist(),
                )
            print('layer3', 
                'anchor', model.module.backbone.layer3[0].conv2.depth_anchor.view(5).tolist(), 
                'temperature', model.module.backbone.layer3[0].conv2.temperature.view(1).tolist(),
                'weight', F.softmax(model.module.backbone.layer3[0].conv2.kernel_weight).view(3).tolist(),
                )
            print('layer4', 
                'anchor', model.module.backbone.layer4[0].conv2.depth_anchor.view(5).tolist(), 
                'temperature', model.module.backbone.layer4[0].conv2.temperature.view(1).tolist(),
                'weight', F.softmax(model.module.backbone.layer4[0].conv2.kernel_weight).view(3).tolist(),
                )
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        loss_sum = 0.
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            imgs = minibatch['data']
            gts = minibatch['label']
            hha = minibatch['hha_img']
            depth = minibatch['depth_img']
            coordinate = minibatch['coord_img']
            camera_params = minibatch['camera_params']
            for k1,v1 in camera_params.items():
                if isinstance(v1, dict):
                    for k2, v2 in v1.items():
                        camera_params[k1][k2] = v2.cuda(non_blocking=True)
                else:
                    camera_params[k1] = v1.cuda(non_blocking=True)

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            # hha = hha.cuda(non_blocking=True)
            depth = depth.cuda(non_blocking=True)
            coordinate = coordinate.cuda(non_blocking=True)

            loss = model(imgs, depth, coordinate, camera_params, gts)

            if engine.distributed:
                dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss = loss / engine.world_size
            else:
                loss = Reduce.apply(*loss) / len(loss)
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.param_groups[2]['lr'] = lr#*2
            for i in range(3, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % (loss_sum/(idx + 1)) 

            pbar.set_description(print_str, refresh=False)
            #pbar.set_description(print_str)

        if dist.get_rank()==0:
            writer.add_scalar('loss', loss_sum/(idx + 1), epoch)

        if (epoch > config.nepochs - 10) or (epoch % config.snapshot_iter == 0):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
    if dist.get_rank()==0:
        writer.close()
