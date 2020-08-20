#!/bin/bash
set -e

dataset="iris_seg"
backbone="xception"

CUDA_VISIBLE_DEVICES=0,1 python train.py \
--backbone $backbone \
--lr 7e-3 \
--workers 4 \
--epochs 500 \
--batch-size 4 \
--test-batch-size 4 \
--gpu-ids 0,1 \
--checkname deeplab-$backbone \
--eval-interval 1 \
--dataset $dataset
