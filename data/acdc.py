"""
Taken/modified from:
https://github.com/naver/oasis/blob/master/dataset/acdc_dataset.py
auxiliary files need to be downloaded from the repo

The support for scenes is removed since we do not need them
"""

import os
import numpy as np
import random
import collections
import glob

import torch
import torchvision
from torch.utils import data

from PIL import Image
import cv2

from data.driving_dataset import DrivingDataset

class ACDC(DrivingDataset):
    """
    ACDC dataset
    """

    VALID_CONDS = ['fog', 'night', 'rain', 'snow']

    def __init__(self, root, cond_list, crop_size=(960, 540), wct2_random_style_transfer=False,
                 wct2_nn_style_transfer=False):
        """
            params

                root : str
                    Path to the base_data folder
        """

        self.root = root
        self.crop_size = crop_size
        self.cond_list = cond_list

        self.files = []
        self.label_files = []

        assert not (wct2_random_style_transfer and wct2_nn_style_transfer)

        # NOTE if using style transfer, we assume images have been transformed
        # before. Of course, in a practical application this will happen online.
        if wct2_random_style_transfer:
            self.images_root = '/FOLDER/TO/IMAGES/STYLIZED/WITH/WCT2-RANDOM'
        elif wct2_nn_style_transfer:
            self.images_root = '/FOLDER/TO/IMAGES/STYLIZED/WITH/WCT2-RANDOM'
        else:
            self.images_root = self.root

        for cond in cond_list:
            if cond not in ['clean', 'fog', 'night', 'rain', 'snow']:
                raise ValueError(
                    'Unknown conditions [supported are clean, fog, night, rain, snow]')

        self.num_imgs_per_seq = []

        for cond in self.cond_list:
            # find all png files in all subfolders of self.images_root, 'rgb_anon',
            #                 cond, 'train' regardless of scene
            pngs = glob.glob(os.path.join(
                self.images_root, 'rgb_anon',
                cond, 'train', '*', '*.png'))

            self.img_paths = glob.glob(os.path.join(
                self.images_root, 'rgb_anon',
                cond, 'train', '*', '*.png'))
            self.img_paths += glob.glob(os.path.join(
                self.images_root, 'rgb_anon',
                cond, 'val', '*', '*.png'))
            self.img_paths = sorted(self.img_paths)


            # self.img_paths[0].split('/'):  ['', 'path_to_data' 'datasets', 'acdc', 'rgb_anon', 'night', 'train', 'GOPR0351', 'GOPR0351_frame_000159_rgb_anon.png']
            self.label_img_paths = [os.path.join(self.root, 'gt', cond,
                                                 path.split('/')[-3], path.split('/')[-2],
                                                 path.split('/')[-1].rstrip('_rgb_anon.png') + '_gt_labelIds.png')
                                    for path in self.img_paths]

            print(f'{cond}: {len(self.img_paths)},')

            self.num_imgs_per_seq.append(len(self.img_paths))

            for img_path in sorted(self.img_paths):
                name = img_path.split('/')[-1]
                self.files.append({
                    'img': img_path,  # used path
                    'name': name  # just the end of the path
                })

            for label_img_path in sorted(self.label_img_paths):
                name = label_img_path.split('/')[-1]
                self.label_files.append({
                    'label_img': label_img_path,  # used path
                    'label_name': name  # just the end of the path
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image = Image.open(self.files[index]['img']).convert('RGB')
        name = self.files[index]['name']

        label = Image.open(self.label_files[index]['label_img'])
        label_name = self.label_files[index]['label_name']

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)

        label = cv2.resize(np.array(label), self.crop_size, interpolation=0)

        # re-assign labels to filter out non-used ones
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.CLASS_CONV_DICT.items():
            label_copy[label == k] = v

        size = image.size[::-1]
        image = self.pil2tensor(image)

        return image, label_copy, np.array(size), name


def get_acdc(condition='night'):
    """
    Get ACDC dataset. Train/Val/Test splits are merged.
    :param condition:
    :return:
    """
    root = 'path_to_data/acdc'
    assert condition in ACDC.VALID_CONDS
    dataset = ACDC(root=root, cond_list=[condition])
    return dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for cond in ACDC.VALID_CONDS:
        dataset = get_acdc(cond)
        for i in range(1):
            img, label, _, _ = dataset[i]
            plt.imshow(ACDC.tensor2pil(img))
            plt.axis('off')
            plt.show()
            plt.imshow(ACDC.mask2color(label))
            plt.axis('off')
            plt.show()

