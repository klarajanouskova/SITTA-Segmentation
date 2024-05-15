import math

import timm
from timm.models.efficientnet import efficientnet_b0

from torch import nn
import torch
import segmentation_models_pytorch as smp
from torchvision.transforms.v2 import Lambda
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import Tuple

from copy import deepcopy


from util.eval import iou_loss


class MaskLossNet(nn.Module):
    """
    Segmentation Loss Network that estimates the loss from the segmentation output, with a custom cnn backbone.
    """
    def __init__(self, size=(384, 384)):
        super(MaskLossNet, self).__init__()
        self.input_size = size
        #  conv layers followed by fully connected layers
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 256, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)
        self.output_size = (self.input_size[0] // 2 ** 4, self.input_size[1] // 2 ** 4)
        self.fc1 = nn.Linear(256 * self.output_size[0] * self.output_size[1], 64)
        self.fc2 = nn.Linear(64, 1)
        # 0-1 range loss such as IoU is assumed here
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = x.view(-1, 256 * self.output_size[0] * self.output_size[1])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class MaskLossNetEff(MaskLossNet):
    """
    Segmentation Loss Network that estimates the loss from the segmentation output, with timm efficientnet backbone
    """
    def __init__(self, channels=1, max_size=-1):
        super(MaskLossNetEff, self).__init__()
        self.backbone = efficientnet_b0(pretrained=True, in_chans=channels)
        #      replace classification head with regression head
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, 1)
        self.act = nn.Sigmoid()
        self.resize = ResizeLongestSide(max_size).apply_image_torch if max_size > 0 else None

    def forward(self, x, inference=True):
        if self.resize:
            x = self.resize(x)
        x = self.backbone(x)
        if inference:
            x = self.act(x)
        return x


class MaskLossDiscNet(MaskLossNet):
    """
    Segmentation Discriminator Network that estimates whether the mask comes from training domain or not.
    Same as MaskLossNet for now but targets should be binary.

    It should be trained together with the segmentation model_seg eventually.
    """


class MaskLossUnet(nn.Module):
    """
    Network that predicts a corrected mask from the segmentation output so that segmentation loss is minimized.
    The architecture is a denoising autoencoder
    f
    channels: 1 for class agnostic binary mask refiner, C for class specific binary mask refinement
     (can be used in semantic segmentation)

    """
    def __init__(self, channels=1, max_size=-1):
        super(MaskLossUnet, self).__init__()
        self.autoenc = smp.Unet(
            encoder_name="efficientnet-b0",  # choose a small encoder
            encoder_weights="imagenet",  # use `imagenet` pre-trained weighgit ts for encoder initialization
            in_channels=channels,  # model_seg input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=channels,  # model_seg output channels (number of classes in your dataset)
            encoder_depth=5 # default and max is 5
        )
        # TODO make this CE when using multiple classes
        self.act = nn.Sigmoid() if channels == 1 else nn.Softmax(dim=1)
        self.resize = ResizeLongestSide(max_size).apply_image_torch if max_size > 0 else None

    def forward(self, x, inference=True):
        orig_shape = x.shape[-2:]
        if self.resize:
            x = self.resize(x)
        #     resize x to be a multiple of 32
        new_shape = [int(math.ceil(s / 32) * 32) for s in x.shape[-2:]]
        x = F.interpolate(x, size=new_shape, mode='bilinear')

        x = self.autoenc(x)
        if inference:
            x = self.act(x)

        #     resize back to original shape
        x = F.interpolate(x, size=orig_shape, mode='bilinear')
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


# copied from segment_anything.utils.transforms to avoid more dependencies
class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)



def test_loss_prediction():
    im, mask, mask_pred = torch.randn(2, 3, 384, 384), (torch.randn(2, 1, 384, 384) > 0.5).float(), (torch.randn(2, 1, 384, 384) > 0.5).float()

    model_loss = MaskLossNet()
    mse = nn.MSELoss()

    summary(model_loss, (1, 384, 384))
    loss_pred = model_loss(mask_pred)
    iou = iou_loss(mask_pred, mask, reduction='none')
    loss = mse(iou, loss_pred)
    print()


def test_pred_discriminator():
    im, mask, mask_pred = torch.randn(2, 3, 384, 384), (torch.randn(2, 1, 384, 384) > 0.5).float(), (torch.randn(2, 1, 384, 384) > 0.5).float()

    model_disc = MaskLossDiscNet()
    mse = nn.MSELoss()

    summary(model_disc, (1, 384, 384))
    loss_pred = model_disc(mask_pred)
    iou = iou_loss(mask_pred, mask, reduction='none')
    loss = mse(iou, loss_pred)
    print()


def test_mask_correction():
    im, mask, mask_pred = torch.randn(2, 3, 384, 384), (torch.randn(2, 1, 384, 384) > 0.5).float(), (torch.randn(2, 1, 384, 384) > 0.5).float()

    model_corr = MaskLossUnet()
    mse = nn.MSELoss()

    summary(model_corr, (1, 384, 384))
    mask_corr = model_corr(mask_pred)
    loss = mse(mask_corr, mask)
    print()


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    # test_loss_prediction()
    # test_pred_discriminator()
    test_mask_correction()

