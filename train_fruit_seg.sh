#!/bin/bash
set -e

dataset="fruit_seg"
backbone="mobilenet"

CUDA_VISIBLE_DEVICES=0,1 python train.py \
--backbone $backbone \
--lr 7e-3 \
--workers 4 \
--epochs 60 \
--batch-size 16 \
--gpu-ids 0,1 \
--checkname deeplab-$backbone \
--eval-interval 1 \
--dataset $dataset