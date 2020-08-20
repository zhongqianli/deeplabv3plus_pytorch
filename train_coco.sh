#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0,1 python train.py \
--backbone resnet \
--lr 0.01 \
--workers 4 \
--epochs 10000 \
--batch-size 16 \
--gpu-ids 0,1 \
--checkname deeplab-resnet \
--eval-interval 1 \
--dataset coco
