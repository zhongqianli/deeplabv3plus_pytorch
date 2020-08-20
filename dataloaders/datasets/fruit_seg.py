from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
# from dataloaders import custom_transforms as tr

import numpy as np
from PIL import Image
import cv2
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch

from mypath import Path


class FruitSegmentation(Dataset):
    NUM_CLASSES = 6
    def __init__(self, base_dir=Path.db_root_dir('fruit_seg'), split="train"):
        super().__init__()
        self.split = split
        with open(r"{}/{}.txt".format(base_dir, split), "r") as f:
            self.lines = f.readlines()

        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'jpg')
        self._cat_dir = os.path.join(self._base_dir, 'png')

        self.images = []
        self.categories = []

        for line in self.lines:
            line = line.strip("\n").strip()
            _image = os.path.join(self._image_dir, line)
            _cat = os.path.join(self._cat_dir, line)
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.images.append(_image)
            self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        HEIGHT = 513
        WIDTH = 513

        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)

        _img, _target = self._make_img_gt_point_pair(index)
        _img = _img.resize((WIDTH, HEIGHT), Image.NEAREST)
        _target = _target.resize((WIDTH, HEIGHT), Image.NEAREST)

        _img = transforms.ToTensor()(_img)
        mean = (0.5345, 0.5792, 0.5067)
        std = (0.1456, 0.1710, 0.1572)
        _img = transforms.Normalize(mean=mean, std=std)(_img)

        _target = np.array(_target).astype(np.float32)
        _target = torch.from_numpy(_target).float()

        sample = {'image': _img, 'label': _target}

        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index]).convert("P")
        return _img, _target


def build_dataloader(root_dir):
    train_dataset = FruitSegmentation(root_dir, split='train')
    test_dataset = FruitSegmentation(root_dir, split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    root_dir = "data/imgseg/dataset-494"

    train_dataloader, test_dataloader = build_dataloader(root_dir)

    for ii, sample in enumerate(train_dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'][jj]
            gt = sample['label'][jj]
            gt_data = gt.numpy()
            gt_set = set(np.round(gt_data.flatten()).astype(np.int32))
            print("gt_set: {}".format(gt_set))

            grid = torchvision.utils.make_grid(img)
            np_imgs = grid.numpy()
            np_imgs = np_imgs.transpose((1, 2, 0))
            plt.imshow(np_imgs)
            plt.show()

            grid = torchvision.utils.make_grid(gt)
            np_imgs = grid.numpy()
            np_imgs = np_imgs.transpose((1, 2, 0))
            plt.imshow(np_imgs)
            plt.show()
            break

        if ii == 1:
            break

    plt.show(block=True)