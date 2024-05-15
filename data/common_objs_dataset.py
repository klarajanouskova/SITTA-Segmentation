'''This code is adapted from https://github.com/Tramac/awesome-semantic-segmentation-pytorch/tree/master'''

import os
import pickle
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils import data

from PIL import Image, ImageOps, ImageFilter

class CommonObjectsDataset(data.Dataset):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    PALETTE_PIL = [128, 64, 128,
                   244, 35, 232,
                   70, 70, 70,
                   102, 102, 156,
                   190, 153, 153,
                   153, 153, 153,
                   250, 170, 30,
                   220, 220, 0,
                   107, 142, 35,
                   152, 251, 152,
                   70, 130, 180,
                   220, 20, 60,
                   255, 0, 0,
                   0, 0, 142,
                   0, 0, 70,
                   0, 60, 100,
                   0, 80, 100,
                   0, 0, 230,
                   119, 11, 32]

    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]
    CATEGORY_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    # COCOCAT2NAME = {cat: CATEGORY_NAMES[cat] for cat in CAT_LIST}
    # VOC2COCO_CAT = {voc_cat: coco_cat for voc_cat, coco_cat in enumerate(CAT_LIST)}
    # VOC2CATNAME = {voc_cat: CATEGORY_NAMES[coco_cat] for voc_cat, coco_cat in enumerate(CAT_LIST)}
    NUM_CLASS = 21

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(CommonObjectsDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @staticmethod
    def mask2color(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(CommonObjectsDataset.PALETTE_PIL)
        return new_mask

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0

    @staticmethod
    def np2pil(image):
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        return image

    @staticmethod
    def pil2tensor(image):
        # pil to np, scale to [0, 1] range
        image = np.asarray(image, np.float32) / 255
        # normalize
        image = CommonObjectsDataset.normalize(image)
        # hwc to chw
        #     #         lambda x: torch.permute(x, (1, 2, 0))
        image = np.transpose(image, (2, 0, 1))
        # np to torch
        image = torch.from_numpy(image.copy()).float()
        return image

    @staticmethod
    def tensor2pil(image):
        # tensor to np
        image = image.cpu().numpy()
        # denormalize
        image = CommonObjectsDataset.denormalize(image)
        # chw to hwc
        image = np.transpose(image, (1, 2, 0))
        # convert to 0-255
        image = image * 255
        # np to pil
        image = CommonObjectsDataset.np2pil(image)
        return image

    @staticmethod
    def normalize(im):
        if isinstance(im, torch.Tensor):
            #  if im is a tensor, convert mean from np to tensor
            mean = torch.from_numpy(CommonObjectsDataset.mean).float().to(im.device)
            std = torch.from_numpy(CommonObjectsDataset.std).float().to(im.device)
            # b x 3 x h x w
            im = (im - mean[:, None, None]) / std[:, None, None]
        else:
            # im is np array
            im = (im - CommonObjectsDataset.mean) / CommonObjectsDataset.std
        return im

    @staticmethod
    def denormalize(im):
        if isinstance(im, torch.Tensor):
            #  if im is a tensor, convert mean from np to tensor
            mean = torch.from_numpy(CommonObjectsDataset.mean).float().to(im.device)
            std = torch.from_numpy(CommonObjectsDataset.std).float().to(im.device)
            # b x 3 x h x w
            im = im * std[:, None, None] + mean[:, None, None]
        else:
            # im is np array
            im = im * CommonObjectsDataset.std + CommonObjectsDataset.mean
        return im


    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')