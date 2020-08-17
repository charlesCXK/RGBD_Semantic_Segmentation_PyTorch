#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from utils.img_utils import pad_image_to_shape, normalize
from engine.evaluator import Evaluator
from engine.logger import get_logger
from seg_opr.metric import hist_info, compute_score
from nyu import NYUv2
from dataloader import ValPre
from network import DeepLab

logger = get_logger()


class SegEvaluator(Evaluator):
    def sliding_eval_rgbd_coord(self, img, hha, depth, coord, camera_params, crop_size, stride_rate, device=None):
        ori_rows, ori_cols, c = img.shape
        processed_pred = np.zeros((ori_rows, ori_cols, self.class_num))

        for s in self.multi_scales:
            img_scale = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            hha_scale = cv2.resize(hha, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_LINEAR)
            depth_scale = cv2.resize(depth, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_NEAREST)
            coord_scale = cv2.resize(coord, None, fx=s, fy=s,
                                    interpolation=cv2.INTER_NEAREST)
            camera_params['scale'] = torch.from_numpy(np.array(s, dtype=np.float32)).float() 
            new_rows, new_cols, _ = img_scale.shape
            processed_pred += self.scale_process_rgbd_coord(img_scale, hha_scale, depth_scale, coord_scale, 
                                                 camera_params,
                                                 (ori_rows, ori_cols),
                                                 crop_size, stride_rate, device)

        pred = processed_pred.argmax(2)

        return pred

    def scale_process_rgbd_coord(self, img, hha, depth, coord, camera_params, ori_shape, crop_size, stride_rate, device=None):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if long_size <= crop_size:
            input_data, input_hha, input_depth, input_coord, margin = self.process_image_rgbd_coord(img, hha, depth, coord, crop_size)
            score = self.val_func_process_rgbd_coord(input_data, input_hha, input_depth, input_coord, camera_params, device)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride = int(np.ceil(crop_size * stride_rate))
            img_pad, margin = pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)
            hha_pad, margin = pad_image_to_shape(hha, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)
            depth_pad, margin = pad_image_to_shape(depth, crop_size,
                                                  cv2.BORDER_CONSTANT, value=0)
            coord_pad, margin = pad_image_to_shape(coord, crop_size,
                                                  cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size) / stride)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size) / stride)) + 1
            data_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)
            count_scale = torch.zeros(self.class_num, pad_rows, pad_cols).cuda(
                device)

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride
                    s_y = grid_yidx * stride
                    e_x = min(s_x + crop_size, pad_cols)
                    e_y = min(s_y + crop_size, pad_rows)
                    s_x = e_x - crop_size
                    s_y = e_y - crop_size
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    hha_sub = hha_pad[s_y:e_y, s_x: e_x, :]
                    depth_sub = depth_pad[s_y:e_y, s_x: e_x]
                    coord_sub = coord_pad[s_y:e_y, s_x: e_x]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, input_hha, input_depth, input_coord, tmargin = self.process_image_rgbd_coord(img_sub, hha_sub, depth_sub, coord_sub, crop_size)
                    temp_score = self.val_func_process_rgbd_coord(input_data, input_hha, input_depth, input_coord, camera_params, device)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            score = data_scale #/ count_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output
    
    def process_image_rgbd_coord(self, img, hha, depth, coord, crop_size=None):
        p_img = img
        p_hha = hha
        p_depth = depth
        p_coord = coord

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        p_img = normalize(p_img, self.image_mean, self.image_std)
        # p_depth = normalize(p_depth, 0, 1)
        

        if crop_size is not None:
            p_img, margin = pad_image_to_shape(p_img, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_hha, margin = pad_image_to_shape(p_hha, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_depth, margin = pad_image_to_shape(p_depth, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
            p_coord, margin = pad_image_to_shape(p_coord, crop_size,
                                               cv2.BORDER_CONSTANT, value=0)
        p_img = p_img.transpose(2, 0, 1)
        p_hha = p_hha.transpose(2, 0, 1)
        p_depth = p_depth[np.newaxis,...]
        p_coord = p_coord.transpose(2, 0, 1)

        return p_img, p_hha, p_depth, p_coord, margin
    
    def val_func_process_rgbd_coord(self, input_data, input_hha, input_depth, input_coord, camera_params, device=None):
        input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                          dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda(device)
        
        input_hha = np.ascontiguousarray(input_hha[None, :, :, :],
                                          dtype=np.float32)
        input_hha = torch.FloatTensor(input_hha).cuda(device)
        
        input_depth = np.ascontiguousarray(input_depth[None, :, :, :],
                                          dtype=np.float32)
        input_depth = torch.FloatTensor(input_depth).cuda(device)

        input_coord = np.ascontiguousarray(input_coord[None, :, :, :],
                                          dtype=np.float32)
        input_coord = torch.FloatTensor(input_coord).cuda(device)
        for k1,v1 in camera_params.items():
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    camera_params[k1][k2] = v2.view(1,).cuda(device)
            else:
                camera_params[k1] = v1.view(1,).cuda(device)

        with torch.cuda.device(input_data.get_device()):
            self.val_func.eval()
            self.val_func.to(input_data.get_device())
            with torch.no_grad():
                score = self.val_func(input_data, input_depth, input_coord, camera_params)
                score = score[0]
                counter = 1

                if self.is_flip:
                    input_data = input_data.flip(-1)
                    input_hha = input_hha.flip(-1)
                    input_depth = input_depth.flip(-1)
                    input_coord = input_coord.flip(-1)
                    score_flip = self.val_func(input_data, input_depth, input_coord, camera_params)
                    score_flip = score_flip[0]
                    score += score_flip.flip(-1)
                    counter += 1
                # score = torch.exp(score)
                # score = score.data
                score /= counter
                score = F.softmax(score, dim=0)

        return score

    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        hha = data['hha_img']
        depth = data['depth_img']
        coord = data['coord_img']
        camera_params = data['camera_params']
        pred = self.sliding_eval_rgbd_coord(img, hha, depth, coord, camera_params, config.eval_crop_size,
                                 config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes,
                                                       pred,
                                                       label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}

        if self.save_path is not None:
            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)
        print(len(dataset.get_class_names()))
        result_line = print_iou(iu, mean_pixel_acc,
                                dataset.get_class_names(), True)
        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = DeepLab(config.num_classes, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'hha_root': config.hha_root_folder,
                    'depth_root':config.depth_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    val_process = ValPre()
    dataset = NYUv2(data_setting, 'val', val_process)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
