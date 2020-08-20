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
import sys

sys.path.append("./")
sys.path.append("./modeling")

import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from modeling.deeplab import *


def voc_classes():
    return ["background",
            "apple",
            "kiwifruit",
            "lemon",
            "mango",
            "pear"
            ]


colors = [
    (0, 0, 0),  # 黑色， 背景
    (128, 128, 128),  # 灰色， 苹果
    (255, 255, 0),  # 青色， 猕猴桃
    (255, 0, 0),  # 蓝色， 柠檬
    (255, 255, 255),  # 白色， 黄芒果
    (255, 255, 0),  # 红色， 梨
]


def foreach_files(root_dir):
    filenames = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".bmp") or file.endswith(".png"):
                filename = os.path.join(root, file)
                filenames.append(filename)
    return filenames


def main():
    model_file = "run/fruit_seg/deeplab-mobilenet/model_best.pth.tar"
    image_dir = "images/fruit"

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLab(backbone='mobilenet', num_classes=8, output_stride=8)

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    print('==> Loading %s model file: %s' % (model.__class__.__name__, model_file))
    model_data = torch.load(model_file)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['state_dict'])

    model = model.to(device)

    model.eval()


    filenames = foreach_files(image_dir)

    for filename in filenames:
        print(filename)
        img = Image.open(filename).convert("RGB")
        np_img = np.array(img, dtype=np.uint8)
        # np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        img = transforms.ToTensor()(img)
        mean = (0.5345, 0.5792, 0.5067)
        std = (0.1456, 0.1710, 0.1572)
        img = transforms.Normalize(mean=mean, std=std)(img).unsqueeze(dim=0)
        img = img.to(device)

        with torch.no_grad():
            score = model(img)

        score = score.to("cpu")
        result = score.argmax(dim=1).squeeze(dim=0).numpy()

        res = np_img
        mask = res.copy()

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

                mask[row, col, 0] = colors[idx][0]
                mask[row, col, 1] = colors[idx][1]
                mask[row, col, 2] = colors[idx][2]

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
            # cv2.putText(res, "{}".format(label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        cv2.imwrite(filename + ".res.jpg", res)
        plt.imshow(res)
        plt.show()

        cv2.imwrite(filename + ".mask.jpg", mask)
        plt.imshow(mask)
        plt.show()


if __name__ == '__main__':
    main()
