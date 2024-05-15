"""
Taken/modified from:
https://github.com/naver/oasis/blob/master/dataset/cityscapes_dataset.py
auxiliary files need to be downloaded from the repo
"""

import os

import matplotlib.pyplot as plt
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


local = not torch.cuda.is_available()

CITIES_VAL = ['frankfurt', 'lindau', 'munster']
CITIES_TRAIN = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
CITIES_TEST = ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']

split2cities = {'train': CITIES_TRAIN, 'val': CITIES_VAL, 'test': CITIES_TEST}


class Cityscapes(DrivingDataset):
    def __init__(self, root, city_list, cond_list, crop_size=(1024, 512),
                 set='val', alpha=0.02, beta=0.01,
                 dropsize=0.005, pattern=3, wct2_random_style_transfer=False,
                 wct2_nn_style_transfer=False):
        """
            params

                root : str
                    Path to the base_data folder'

        """

        self.root = root
        self.crop_size = crop_size
        self.city_list = city_list
        self.cond_list = cond_list
        self.alpha = alpha
        self.beta = beta
        self.dropsize = dropsize
        self.pattern = pattern

        self.files = []
        self.label_files = []

        for cond in cond_list:
            if cond not in ['clean', 'fog', 'rain']:
                raise ValueError(
                    'Unknown conditions [supported are clean, rain, fog]')

        assert len(cond_list) == len(city_list)

        self.num_imgs_per_seq = []

        assert not (wct2_random_style_transfer and wct2_nn_style_transfer)

        # NOTE if using style transfer, we assume images have been transformed
        # before. Of course, in a practical application this will happen online.
        if wct2_random_style_transfer:
            self.images_root = '/FOLDER/TO/IMAGES/STYLIZED/WITH/WCT2-RANDOM'
        elif wct2_nn_style_transfer:
            self.images_root = '/FOLDER/TO/IMAGES/STYLIZED/WITH/WCT2-RANDOM'
        else:
            self.images_root = self.root

        for city, cond in zip(self.city_list, self.cond_list):
            if city in ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']:
                self.set = 'test'
            elif city in ['frankfurt', 'lindau', 'munster']:
                self.set = 'val'
            else:
                self.set = 'train'

            # Path to the txt containing the relative paths
            # (with respect to root) of the images/labels to load
            list_of_images_file = os.path.join(root, f'cityscapes_list/images_{city}.txt')
            list_of_label_images_file = os.path.join(root, f'cityscapes_list/labels_{city}.txt')

            self.img_names = [i_id.strip() for i_id in open(list_of_images_file)]
            if cond == 'clean':
                pass
            elif cond == 'fog':
                self.img_names = [i_id.rstrip('.png') + f'_foggy_beta_0.02.png' for i_id in self.img_names]
            elif cond == 'rain':
                self.img_names = sorted(glob.glob(os.path.join(
                    self.images_root,
                    f'leftImg8bit_rain/{self.set}/{city}',
                    f'*_alpha_{self.alpha}_beta_{self.beta}_dropsize_{self.dropsize}_pattern_{self.pattern}.png')))
            else:
                raise ValueError('Unknown conditions [supported are clean,rain,fog]')

            self.label_img_names = [i_id.strip() for i_id in open(list_of_label_images_file)]

            if cond == 'rain':
                img_names_ = [_.split(f'/{self.set}/')[1].split('_leftImg8bit_')[0] for _ in self.img_names]
                self.label_img_names = [_ for _ in self.label_img_names if
                                        _.rstrip('_gtFine_labelIds.png') in img_names_]

            print(f'\'{city}\': {len(self.img_names)},')

            self.num_imgs_per_seq.append(len(self.img_names))

            for name in sorted(self.img_names):
                if cond == 'clean':
                    img_path = os.path.join(self.images_root, f'leftImg8bit/{self.set}/{name}')
                elif cond == 'fog':
                    img_path = os.path.join(self.images_root, f'leftImg8bit_foggyDBF/{self.set}/{name}')
                elif cond == 'rain':
                    img_path, name = name, name.split('/')[-1]
                else:
                    raise ValueError('Unknown conditions [supported are clean,rain,fog]')

                self.files.append({
                    'img': img_path,  # used path
                    'name': name  # just the end of the path
                })

            for name in sorted(self.label_img_names):
                img_path = os.path.join(self.root, f'gtFine/{self.set}/{name}')
                self.label_files.append({
                    'label_img': img_path,  # used path
                    'label_name': name  # just the end of the path
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image = Image.open(self.files[index]['img']).convert('RGB')
        name = self.files[index]['name']

        label = Image.open(self.label_files[index]['label_img'])  # .convert('RGB')
        label_name = self.label_files[index]['label_name']

        # resizeimage
        image = image.resize(self.crop_size, Image.BICUBIC)

        # resize label
        label = label.resize(self.crop_size, Image.NEAREST)

        # re-assign labels to filter out non-used ones
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.CLASS_CONV_DICT.items():
            label_copy[label == k] = v

        size = image.size[::-1]
        image = self.pil2tensor(image)

        return image, label_copy, np.array(size), name


class CityscapesSub(Cityscapes):
    """
    Selects only n first samples of each city-cond combination
    """
    def __init__(self, root, city_list, cond_list, n=10, **kwargs):
        super().__init__(root, city_list, cond_list, **kwargs)
        self.n = n
        # randomize the order of the files and pick first n
        self.files = random.shuffle(self.files)[:n]
        self.label_files = random.shuffle(self.label_files) [:n]
        self.num_imgs_per_seq = [n] * len(self.city_list)

    def __len__(self):
        return len(self.files)


def get_cityscapes(split='train', condition='clean'):
    """
    :return: Cityscapes dataset
    """
    assert split in ['train', 'val', 'test']
    assert condition in ['clean', 'fog', 'rain']
    root = os.path.join('path_to_data', 'cityscapes')
    cities = split2cities[split]
    cond_list = [condition] * len(cities)
    return Cityscapes(root=root, city_list=cities, cond_list=cond_list)




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    root = os.path.join('path_to_data', 'cityscapes')
    cities = CITIES_VAL[:2]
    cond_list = ['clean'] * len(cities)
    dataset = Cityscapes(root=root, city_list=cities, cond_list=cond_list)

    for i in range(10):
        img, label, _, _ = dataset[i]
        img = img.transpose((1, 2, 0))
        img += dataset.mean
        img = img[:, :, ::-1]
        img = np.asarray(img, np.uint8)
        label = np.asarray(label, np.uint8)
        vis_label = dataset.visualize_mask(label)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        plt.imshow(vis_label, alpha=0.5)
        plt.axis('off')
        plt.show()
