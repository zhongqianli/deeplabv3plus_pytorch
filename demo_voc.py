#!/usr/bin/env python

import argparse
import os
import os.path as osp

import numpy as np
import skimage.io
import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable

import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from modeling.deeplab import *


def voc_classes():
    return ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
            'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
            ]


colors = []
for b in [0, 128, 255]:
    for g in [0, 128, 255]:
        for r in [0, 128, 255]:
            colors.append((b, g, r))


def transform(img):
    # img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img /= 255.0
    img -= mean
    img /= std
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    model_file = "run/pascal/deeplab-resnet/model_best.pth.tar"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define network
    # model = DeepLab(num_classes=21,
    #                 backbone="resnet",
    #                 output_stride=16,
    #                 sync_bn=False,
    #                 freeze_bn=False)

    model = DeepLab(backbone='resnet', output_stride=16)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['state_dict'])

    model = model.to(device)

    model.eval()

    filename = "images/2007_000129.jpg"
    filename = "images/2007_000033.jpg"
    filename = "images/2007_000364.jpg"
    filename = "images/004481.jpg"
    filename = "images/car.jpg"
    filename = "images/car1.jpg"

    img = Image.open(filename).convert("RGB")
    img = np.array(img, dtype=np.uint8)
    img_copy = img.copy()
    img = transform(img)


    # img = torchvision.transforms.ToTensor()(img)
    img = img.unsqueeze(dim=0)
    img = img.to(device)

    with torch.no_grad():
        score = model(img)

    score = score.to("cpu")
    result = score.argmax(dim=1).squeeze(dim=0).numpy()

    img = img.to("cpu")
    img = img.squeeze(dim=0).numpy()
    img = np.transpose(img, (1, 2, 0)) * 255
    img = img.astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    plt.imshow(img)
    plt.show()

    res = img_copy

    label_dict = dict()
    for label in voc_classes():
        label_dict[label] = dict(x=0, y=0, n=0)

    rows, cols = np.shape(result)
    for row in range(rows):
        for col in range(cols):
            idx = result[row, col]

            alpha = 0.5
            res[row, col, 0] = alpha * colors[idx][0] + (1 - alpha) * res[row, col, 0]
            res[row, col, 1] = alpha * colors[idx][1] + (1 - alpha) * res[row, col, 1]
            res[row, col, 2] = alpha * colors[idx][2] + (1 - alpha) * res[row, col, 2]

            x = label_dict[voc_classes()[idx]]["x"] + col
            y = label_dict[voc_classes()[idx]]["y"] + row
            n = label_dict[voc_classes()[idx]]["n"] + 1
            label = voc_classes()[idx]
            label_dict[label] = dict(x=x, y=y, n=n)

    for label in voc_classes():
        n = label_dict[label]["n"]
        if n == 0:
            continue
        x = int(label_dict[label]["x"] / n)
        y = int(label_dict[label]["y"] / n)
        print("label:{}".format(label))
        cv2.putText(res, "{}".format(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    plt.imshow(res)
    plt.show()


if __name__ == '__main__':
    main()
