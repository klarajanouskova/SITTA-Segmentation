import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from data.coco import COCOSegmentation
from data.voc import PALETTE_PIL as VOC_PALETTE
from data.gta5 import GTA5

class CValDataset(Dataset):
    """
    Corrupted validation dataset
    """
    def __init__(self, root, base_root, base_name, corruption, severity, n_samples=40, transform=None):
        assert base_name in ['gta5', 'coco'], f'base_name should be gta5 or coco, got {base_name}'
        # self.root is already used by GTA5/COCO datasets so it would get overwritten
        self.corr_root = root
        self.base_root = base_root
        self.corruption = corruption
        self.severity = severity
        self.n_samples = n_samples
        self.transform = transform
        self.base_name = base_name

    @property
    def img_dir(self):
        return os.path.join(self.corr_root, 'images')

    @property
    def gt_dir(self):
        return os.path.join(self.corr_root, 'labels')

    def idx_to_label(self, idx):
        path = os.path.join(self.gt_dir, f'{idx}.png')
        label = Image.open(path)
        return label, path

    def idx_to_corr_img(self, idx):
        if self.corruption == 'none':
            return self.idx_to_orig_img(idx)
        img_file_name = f'{self.corruption}_s{self.severity}_{idx}.png'
        path = os.path.join(self.img_dir, img_file_name)
        img = Image.open(path).convert('RGB')
        return img, path

    def idx_to_orig_img(self, idx):
        path = os.path.join(self.img_dir, f'{idx}.png')
        image = Image.open(path).convert('RGB')
        return image, path

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        img, _ = self.idx_to_corr_img(index)
        gt, _ = self.idx_to_label(index)
        name = self.idx_to_name(index)
        orig_img, _ = self.idx_to_orig_img(index)
        if self.transform is not None:
            img = self.transform(img)
            orig_img = self.transform(orig_img)
            gt = self.transform(gt)
        return img, gt, orig_img, name


class GTA5CValDataset(CValDataset, GTA5):

    PALETTE_PIL = GTA5.PALETTE_PIL

    def __init__(self, root, base_root, base_name, corruption, severity, n_samples=40, transform=None):
        CValDataset.__init__(self, root, base_root, base_name, corruption, severity, n_samples, transform)
        GTA5.__init__(self, root=base_root, split='val', valid_size=250)

    def idx_to_name(self, idx):
        name = self.files[idx]['name']
        return name

    @staticmethod
    def mask2color(mask):
        # copy the mask
        mask = mask.convert('P')
        mask.putpalette(GTA5CValDataset.PALETTE_PIL)
        return mask



class COCOCValDataset(CValDataset, COCOSegmentation):

    PALETTE_PIL = COCOSegmentation.PALETTE_PIL

    def __init__(self, root, base_root, base_name, corruption, severity, n_samples=40, transform=None):
        CValDataset.__init__(self, root, base_root, base_name, corruption, severity, n_samples, transform)
        COCOSegmentation.__init__(self, root=base_root, split='val', transform=None)
    def idx_to_name(self, idx):
        name = self.coco.loadImgs(self.ids[idx])[0]['file_name']
        return name

    @staticmethod
    def mask2color(mask):
        # copy the mask
        mask = mask.convert('P')
        mask.putpalette(VOC_PALETTE)
        return mask

def test_gta5():
    import numpy as np
    np.random.seed(0)

    base_root = '/datagrid/personal/janoukl1/datasets/gta5/'
    root = '/datagrid/personal/janoukl1/datasets/TTA/gta5_corr'
    dataset = GTA5CValDataset(root=root, base_root=base_root, base_name='gta5', corruption='fog', severity=5)
    print(len(dataset))
    img, gt, orig_img, name = dataset[0]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[1].imshow(dataset.mask2color(gt))
    axs[2].imshow(orig_img)
    plt.show()


def test_coco():
    import numpy as np
    np.random.seed(0)

    base_root = '/datagrid/personal/janoukl1/datasets/COCO/'
    root = '/datagrid/personal/janoukl1/datasets/TTA/coco_corr'
    dataset = COCOCValDataset(root=root, base_root=base_root, base_name='coco', corruption='fog', severity=5)
    print(len(dataset))
    img, gt, orig_img, name = dataset[0]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[1].imshow(dataset.mask2color(gt))
    axs[2].imshow(orig_img)
    plt.show()


if __name__ == '__main__':
    test_coco()