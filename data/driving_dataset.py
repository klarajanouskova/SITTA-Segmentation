import torch
from torch.utils import data
from PIL import Image
import numpy as np

# doesn't need to be a class, it can be a module
class DrivingDataset(data.Dataset):
    """
    Dataset class for the driving datasets - GTA5, Cityscapes, ACDC
    Contains normalization and visualization functions
    """
    mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

    CLASS_CONV_DICT = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
        23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

    CLASS2CAT = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole',
                 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain', 10: 'sky',
                 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train',
                 17: 'motorcycle', 18: 'bicycle'}

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    PALETTE_PIL = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
                153, 153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152,
                70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
                0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]


    @staticmethod
    def visualize_mask(mask):
        """

        :param mask:
        :return:
        """
        mask = np.asarray(mask, np.float32)
        mask = np.expand_dims(mask, axis=2)
        mask = np.tile(mask, (1, 1, 3))
        return mask

    @staticmethod
    def mask2color(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(DrivingDataset.PALETTE_PIL)
        return new_mask

    @staticmethod
    def tensor2pil(image):
        # torch to np
        image = image.cpu().numpy().squeeze()
        # chw to hwc
        image = np.transpose(image, (1, 2, 0))
        # denormalize
        image = DrivingDataset.denormalize(image)
        # bgr to rgb
        image = image[:, :, ::-1]
        # np to pil
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        return image

    @staticmethod
    def pil2tensor(image):
        # pil to np, no rescaling
        image = np.asarray(image, np.float32)
        # rgb to bgr
        image = image[:, :, ::-1]
        # normalize
        image = DrivingDataset.normalize(image)
        # hwc to chw
        image = np.transpose(image, (2, 0, 1))
        # np to torch
        image = torch.from_numpy(image.copy()).float()
        return image

    @staticmethod
    def normalize(im):
        #  if im is a tensor, convert mean to tensor
        if isinstance(im, torch.Tensor):
            mean = torch.from_numpy(DrivingDataset.mean.copy()).float().to(im.device)
            # b x 3 x h x w
            im = im - mean[:, None, None]
        else:
            im = im - DrivingDataset.mean
        return im

    @staticmethod
    def denormalize(im):
        #  if im is a tensor, convert mean to tensor
        if isinstance(im, torch.Tensor):
            mean = torch.from_numpy(DrivingDataset.mean.copy()).float().to(im.device)
            # b x 3 x h x w
            im = im + mean[:, None, None]
        else:
            im = im + DrivingDataset.mean
        return im


