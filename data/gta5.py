"""
Taken/modified from:
https://github.com/naver/oasis/blob/master/dataset/gta5_dataset.py
auxiliary files need to be downloaded from the repo
"""

import os
import numpy as np
import numpy.random as npr
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import copy

from data.driving_dataset import DrivingDataset


class GTA5(DrivingDataset):
    def __init__(self, root, num_epochs=1, crop_size=(1280, 720), split='train', valid_size=250):
        self.root = root
        self.list_path = os.path.join(root, 'gta5_list/train.txt')
        self.crop_size = crop_size
        self.img_names_ = [i_id.strip() for i_id in open(self.list_path)]  # [:10]
        total_n = len(self.img_names_)
        if split == 'train':
            self.img_names_ = self.img_names_[:total_n - valid_size]
        elif split == 'val':
            self.img_names_ = self.img_names_[total_n - valid_size:]
        self.img_names = []
        # keeping the same np random seed is important for reproducibility here!
        if num_epochs is not None:
            img_names_ = copy.deepcopy(self.img_names_)
            for _ in range(num_epochs):
                npr.shuffle(img_names_)
                self.img_names += img_names_
        else:
            self.img_names = self.img_names_

        self.files = []
        self.class_conversion_dict = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                                      19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                                      26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in self.img_names:
            img_path = os.path.join(self.root, f'images/{name}')
            label_path = os.path.join(self.root, f'labels/{name}')
            self.files.append({
                'img': img_path,
                'label': label_path,
                'name': name
            })

        self.img_files = [_['img'] for _ in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image = Image.open(self.files[index]['img']).convert('RGB')
        label = Image.open(self.files[index]['label'])
        name = self.files[index]['name']

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.class_conversion_dict.items():
            label_copy[label == k] = v

        size = image.size[::-1]
        image = self.pil2tensor(image)

        return image, label_copy, np.array(size), name


def get_gta5(data_path, val_size=250, split='train'):
    root = os.path.join(data_path, 'gta5')
    if split == 'train':
        dataset = GTA5(root, split='train', valid_size=val_size)
    elif split == 'val':
        dataset = GTA5(root, split='val', valid_size=val_size)
    else:
        raise NotImplementedError(f'No "{split}" set for GTA5')
    return dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = get_gta5(split='val')
    #     visualize first 2 samples
    for i in range(2):
        data = dataset[i]
        img = data[0]
        label = data[1]
        name = data[3]
        plt.imshow(dataset.tensor2pil(img))
        plt.show()
        plt.imshow(label)
        plt.show()
