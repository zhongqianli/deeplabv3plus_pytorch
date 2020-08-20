from __future__ import print_function, division
import os
from torchvision import transforms

import numpy as np
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt

from mypath import Path


class IrisSegmentation(Dataset):
    NUM_CLASSES = 2

    def __init__(self, base_dir=Path.db_root_dir('iris_seg'), split="train"):
        super().__init__()
        
        self.img_filenames = []
        self.label_filenames = []

        image_dir = os.path.join(base_dir, split, "image")
        for root, dirs, files in os.walk(image_dir):
            for filename in files:
                if not filename.endswith(".JPEG"):
                    continue
                img_filename = os.path.join(root, filename)
                filename = filename.strip(".JPEG")
                label_filename = os.path.join(base_dir, split, "SegmentationClass/{}.png".format(filename))
                if os.path.exists(img_filename) and os.path.exists(label_filename):
                    self.img_filenames.append(img_filename)
                    self.label_filenames.append(label_filename)

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.img_filenames)))

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        _img = transforms.ToTensor()(_img)

        _target = np.array(_target, dtype=np.int32)
        _target[_target == 255] = 1
        _target = torch.from_numpy(_target).float()

        sample = {'image': _img, 'label': _target}

        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.img_filenames[index]).convert('RGB')
        _target = Image.open(self.label_filenames[index]).convert("P")
        return _img, _target


def build_dataloader(root_dir):
    train_dataset = IrisSegmentation(root_dir, split='train')
    test_dataset = IrisSegmentation(root_dir, split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    root_dir = "data/imgseg/iris_seg/CASIA-distance"

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
