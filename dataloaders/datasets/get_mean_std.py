'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

import torch
import torchvision
from torchvision import transforms
import pandas as pd
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.imagepath_list = self.foreach_images(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, item):
        imagepath = self.imagepath_list[item]
        print("[{}] {}".format(item, imagepath))
        assert os.path.exists(imagepath)
        img = self.read_image(imagepath)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def read_image(self, path):
        from PIL import Image
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def foreach_files(self, root_dir, ext_name):
        g = os.walk(root_dir)
        filename_list = []
        for path, dir_list, file_list in g:
            for file in file_list:
                if len(file) > len(ext_name) and file[-len(ext_name):] == ext_name:
                    filename = os.path.join(path, file)
                    filename = filename.replace('\\', '/')
                    filename_list.append(filename)
        return filename_list


    def foreach_images(self, root_dir):
        filename_list = []
        filename_list = self.foreach_files(root_dir, '.jpg')
        filename_list.extend(self.foreach_files(root_dir, '.bmp'))
        filename_list.extend(self.foreach_files(root_dir, '.png'))
        return filename_list


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print("len(dataloader): ", len(dataloader))
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std...')
    for inputs in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def main():
    root_dir = "data/imgseg/dataset-494/jpg"
    dataset = Dataset(root_dir, transform=transforms.Compose([transforms.ToTensor()]))
    mean, std = get_mean_and_std(dataset)
    print("mean1: {0}, std1: {1}".format(mean, std))
    print("mean2: {0}, std2: {1}".format(mean * 255, std * 255))


if __name__ == '__main__':
    main()