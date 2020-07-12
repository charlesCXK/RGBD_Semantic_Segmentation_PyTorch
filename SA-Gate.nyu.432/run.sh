#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
python eval.py -e 300-400 -d 0-3 --save_path results