from functools import partial

import torch.nn as nn
import torch
import torch.nn.functional as F


def iou_score(pred, gt, thresh=None, apply_sigmoid=False, epsilon=1):
    """
    Computes either soft (thresh=None, default) or hard IoU scores. Assumes that input is * x H x W.
    No reduction is applied.
    :param pred: prediction
    :param gt: ground truth, same shape as prediction
    :param thresh: float, threshold for hard IoU
    :return:
    """
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    if thresh is not None:
        pred = pred > thresh
    #     compute iou, summing over last two dimensions (H, W) with arbitrary number of leading dimensions
    intersection = (pred * gt).sum(dim=(-2, -1))
    union = (pred + gt).sum(dim=(-2, -1)) - intersection
    # avoid zero division by adding epsilon
    union = union + epsilon
    iou = intersection / union
    # print(f'union: {union}, inter: {intersection}, iou: {iou}')
    return iou


def iou_loss(pred, gt, thresh=None, reduction='mean', apply_softmax=False, apply_sigmoid=False, ignore_index=255, epsilon=1):
    """
    Computes either soft (thresh=None, default) or hard IoU loss.
    :param pred: prediction, B x C x H x W
    :param gt: ground truth, same shape as prediction
    :param thresh: float, threshold for hard IoU
    :param reduction: str, 'mean' or 'none'
    :return:
    """
    if apply_softmax:
        pred = pred.softmax(dim=1)
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    #     deal with ignroe index, keeping b x c x h x w shape
    if ignore_index is not None:
        mask = gt == ignore_index
        gt = gt * (~mask)
        pred = pred * (~mask)
    if thresh is not None:
        pred = pred > thresh
    iou = iou_score(pred, gt, epsilon=epsilon)
    # mean over the last dimension
    iou = iou.mean(dim=-1)
    iou_loss = 1 - iou
    if reduction == 'mean':
        iou_loss = iou_loss.mean()
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError
    return iou_loss


def bce_wrapper(pred, mask, reduction='mean', apply_sigmoid=True):
    assert apply_sigmoid is not False, 'F.binary_cross_entropy_with_logits assumed sigmoid was not applied to the output'
    orig = F.binary_cross_entropy_with_logits(pred, mask, reduction=reduction)
    if reduction == 'none':
        # reduce over the spatial dimensions
        orig = orig.mean(dim=(2, 3))
    return orig


class Loss(nn.Module):
    valid_vals = ['STR', 'FL', 'IoU', 'BCE']
    losses_dict = {'BCE': bce_wrapper,
                   'IoU': iou_loss,
                   'IoU_25': partial(iou_loss, threshold=0.25),
                   'IoU_50': partial(iou_loss, threshold=0.5),
                   'IoU_75': partial(iou_loss, threshold=0.75),
                   'IoU_90': partial(iou_loss, threshold=0.9),
                   }

    def __init__(self, loss_str, weights=None, reduction='mean', apply_sigmoid=True):
        self.losses = self.validate_and_parse_loss_str(loss_str)
        if weights is None:
            self.weights = [1] * len(self.losses)
        self.reduction = reduction
        self.apply_sigmoid = apply_sigmoid

    def __call__(self, pred, mask):
        loss_vals = [self.losses_dict[l](pred, mask, reduction=self.reduction, apply_sigmoid=self.apply_sigmoid) for l in self.losses]
        weighted_loss = sum([w * l for w, l in zip(self.weights, loss_vals)])
        loss_vals_dict = {loss: l for loss, l in zip(self.losses, loss_vals)}
        return weighted_loss, loss_vals_dict

    def validate_and_parse_loss_str(self, loss_str):
        parsed_loss = loss_str.split('+')
        for val in parsed_loss:
            if val not in self.valid_vals:
                raise ValueError('Invalid loss function: {}'.format(val))
        return parsed_loss