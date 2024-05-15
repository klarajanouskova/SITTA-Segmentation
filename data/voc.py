'''This code is adapted from https://github.com/Tramac/awesome-semantic-segmentation-pytorch/tree/master'''

from torchvision.datasets import VOCSegmentation
from PIL import Image, ImageOps, ImageFilter
from data.coco import coco_transform, coco_transform_invert
import numpy as np


def get_palette(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


PALETTE = get_palette(20)
# flattened version of PALETTE
PALETTE_PIL = PALETTE.reshape(-1)

CATS = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']

CLASS2CAT = {i: cat for i, cat in enumerate(CATS)}


def mask2color(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(PALETTE_PIL)
    return new_mask


def get_voc(split="train"):
    root = 'path_to_data/voc'
    if split == 'train':
        dataset = VOCSegmentation(root, split='train', transform=coco_transform())
    elif split == 'val':
        dataset = VOCSegmentation(root, image_set='val', transform=coco_transform())
    else:
        raise NotImplementedError(f'No "{split}" set for VOC')
    return dataset



if __name__ == '__main__':
    print(10 * "-")
    print(PALETTE_PIL)