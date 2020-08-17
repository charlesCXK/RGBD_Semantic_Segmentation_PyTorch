#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
python eval.py -e 60-80 -d 0-7