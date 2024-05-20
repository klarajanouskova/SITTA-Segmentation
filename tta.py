from typing import Tuple, Dict

import models_loss

import sys
import os
from functools import partial

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.transforms import RandomResizedCrop, RandAugment
from model.segmenter import DeepLabSegmenter, COCODeepLabSegmenter

import numpy as np
import matplotlib.pyplot as plt

# prevent matpltolib form using scientific notation
plt.rcParams['axes.formatter.useoffset'] = False

from util.misc import load_tta_model
from util.eval import iou_loss

from data.base_dataset import BaseDataset as BaseDatasetClass

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

sys.path.append('')

import gc

local = not torch.cuda.is_available()
device = 'cpu' if local else 'cuda'


def get_seg_model(size, path='ckpts/gta5/dr2_model.pth', base_dataset="gta5"):
    if base_dataset == "coco":
        model = COCODeepLabSegmenter(size=size)
    else:
        model = DeepLabSegmenter(num_classes=19, path=path, size=size)
    return model


def print_cuda_mem_usage(id=0):
    if not local:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(id)
        mem = nvmlDeviceGetMemoryInfo(handle)
        unit_scale = 1024 ** 3
        used = mem.used / unit_scale
        total = mem.total / unit_scale
        print(f'Used: {used:.2f}GB / {total:.2f}GB')


def print_tensors_on_cuda():
    """
    The output is too messy to be of any use, find sth better
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'base_data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


class TestTimeAdaptor:
    """
    TTA for semantic segmentation
    """
    # should contain consistency-based pseudo-labelling as well
    valid_methods = ['tent', 'iou', 'ref', 'pl', 'adv', 'augco']

    def __init__(self, tta_method, cfg, weights=None, mask_ratio=0.75, load_iou=False, base_dataset="gta5",
                 masks_save_path=None):
        self.tta_methods = self.verify_method(tta_method)
        self.method_str = '&'.join(self.tta_methods)
        self.masks_save_path = masks_save_path
        self.base_dataset = base_dataset
        self.num_classes = 21 if self.base_dataset == 'coco' else 19
        self.base_dataset_class = BaseDatasetClass(self.base_dataset)
        if weights is None:
            weights = [1] * len(self.tta_methods)
        self.weights = weights
        assert len(self.tta_methods) == len(self.weights), 'Number of methods and weights must be the same'
        # TODO not used yet
        self.mask_ratio = mask_ratio

        self.segmentation_model = None

        self.cfg = cfg.tta
        self.ckpt_dir = cfg.ckpt_dir
        self.model_path = cfg.base_data.seg_weights

        if 'iou' in self.tta_methods or load_iou:
            self.loss_model = self.load_iou_model()
        if 'ref' in self.tta_methods:
            self.ref_model = self.load_ref_model()
        if 'augco' in self.tta_methods:
            self.color_augment = SubRandAugment(num_ops=1, magnitude=2)

        # string from weights, each weight separated by _
        weights_str = '_'.join([str(w) for w in self.weights])

        self.save_name = f'tta_{self.method_str}_lr_{cfg.tta.lr}_ws_{weights_str}' \
                         f'_its_{cfg.tta.n_iter}_normonly_{int(cfg.tta.train_norm_only)}' \
                         f'_{cfg.tta.optim}_n_im_{cfg.tta.n_ims}_gt_learnt_{cfg.tta.gt_learnt}_loss_{cfg.tta.loss_name}_seed_{cfg.seed}'

    def __call__(self, images, ret_pseudo_masks=False, save_masks=False, gt=None, save_idx=None):
        """
        Assumes tensor shape (N, 3, H, W), normalized
        pseudomasks = False will return empty array
        """

        # reload the weights
        self.segmentation_model = get_seg_model(
            size=(images.shape[-1], images.shape[-2]),
            path=os.path.join(self.ckpt_dir, self.model_path) if self.model_path is not None else None,
            base_dataset=self.base_dataset
        )
        self.segmentation_model.to(device)

        # set eval and freeze
        self.reset_require_grad()

        preds_tta, preds_rec, loss_dict, pseudomasks = self.optimize_ims(
            images,
            num_it=self.cfg.n_iter,
            lr=self.cfg.lr,
            # lr=1e-2,
            debug=False,
            gt=gt,
            ret_pseudo_masks=ret_pseudo_masks,
            save_masks=save_masks,
            save_idx=save_idx
        )

        # take the output from the last optimization step
        xs_tta = preds_tta[:, -1]

        return xs_tta, preds_tta, loss_dict, pseudomasks

    def load_iou_model(self):
        weight_cfg = self.cfg.weights.gta5 if self.base_dataset == 'gta5' else self.cfg.weights.coco
        if self.cfg.gt_learnt:
            weights = weight_cfg.iou.corr.gt if self.cfg.shift_method == "corruption" \
                else weight_cfg.iou.adv.gt
        else:
            weights = weight_cfg.iou.corr.pred if self.cfg.shift_method == "corruption" \
                else weight_cfg.iou.adv.pred
        ckpt_path = os.path.join(self.ckpt_dir, self.base_dataset, weights)
        # cityscapes input size
        # TODO pass this as param
        loss_model = load_tta_model(
            path=ckpt_path,
            model_class=models_loss.MaskLossNetEff,
            num_classes=self.num_classes
        )
        # should only be used for inference
        loss_model.eval()
        loss_model.freeze()
        return loss_model

    def load_ref_model(self):
        weight_cfg = self.cfg.weights.gta5 if self.base_dataset == 'gta5' else self.cfg.weights.coco
        if self.cfg.gt_learnt:
            weights = weight_cfg.ref.corr.gt if self.cfg.shift_method == "corruption" \
                else weight_cfg.ref.adv.gt
        else:
            weights =weight_cfg.ref.corr.pred if self.cfg.shift_method == "corruption" \
                else weight_cfg.ref.adv.pred
        ckpt_path = os.path.join(self.ckpt_dir, self.base_dataset, weights)
        # cityscapes input size
        ref_net = load_tta_model(
            path=ckpt_path,
            model_class=models_loss.MaskLossUnet,
            # TODO remove this param overall since it is not really needed
            num_classes=self.num_classes
        )
        # should only be used for inference
        ref_net.eval()
        ref_net.freeze()
        return ref_net

    def reset_require_grad(self):
        # if 'tent' not in self.tta_methods:
        # makes sure layers like batchnorm or dropout are in eval mode but we can still backprop
        # TODO make this a parameter and do ablation iwth tent
        self.segmentation_model.eval()
        if self.cfg.train_norm_only:
            # we assume tent/adv is the only method so we can freeze stuff here
            self.segmentation_model.train_norm_layers_only()
        else:
            for param in self.segmentation_model.parameters():
                param.requires_grad = True

    def optimize_ims(
            self,
            imgs,
            num_it=20,
            lr=1e-3,
            debug=True,
            # only for debugging purposes
            gt=None,
            optim='sgd',
            momentum=0.9,
            ret_pseudo_masks=False,
            save_masks=False,
            save_idx=None
    ):
        # TODO this should be moved to config valid
        assert optim in ['sgd', 'adam']
        if optim == 'sgd':
            optim = torch.optim.SGD(lr=lr,
                                    params=filter(lambda p: p.requires_grad, self.segmentation_model.parameters()),
                                    momentum=momentum)
        else:
            optim = torch.optim.AdamW(lr=lr,
                                      params=filter(lambda p: p.requires_grad, self.segmentation_model.parameters()))

        preds_seg, preds_rec = [], []
        pseudo_masks = []

        loss_dict = {tta_method: [] for tta_method in self.tta_methods}

        no_tta_pred = None

        for i in range(num_it + 1):
            # segmentation prediction
            losses = []
            optim.zero_grad(set_to_none=True)

            preds_seg_it_raw = self.segmentation_model(imgs, inference=False, upsample=True)
            preds_seg_it = preds_seg_it_raw.softmax(dim=1)
            if self.masks_save_path:
                pred_vis = self.base_dataset_class.mask2color(preds_seg_it[0].argmax(dim=0).cpu().numpy()).convert('RGB')
                pred_vis.save(self.masks_save_path + f"/{save_idx}/it_{i}.png")
                if i == 0:  # save first output as baselines for tta improvement
                    no_tta_pred = preds_seg_it.cpu().detach()[0].argmax(0)

                else:  # perform baseline comparison
                    pred_seg = preds_seg_it.squeeze().argmax(0).cpu().detach()
                    pixel_idxs_better = np.where((pred_seg == gt) & (no_tta_pred != gt))
                    pixel_idxs_worse = np.where((pred_seg != gt) & (no_tta_pred == gt))
                    # pred_seg_error_vis = pred_seg[:, :, None].numpy().repeat(3, axis=2)
                    pred_seg_error_vis = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3))
                    # make improved pixels green
                    pred_seg_error_vis[pixel_idxs_better] = [0, 1, 0]
                    # make worsened pixels red
                    pred_seg_error_vis[pixel_idxs_worse] = [1, 0, 0]
                    # put ignore pixels to black
                    pred_seg_error_vis[gt == 255] = [0, 0, 0]
                    pred_seg_error_vis = pred_seg_error_vis * 255
                    plt.imsave(self.masks_save_path + f"/{save_idx}/it_improve_{i}.jpg",
                               pred_seg_error_vis.astype(np.uint8))

            # visualize preds_seg_it
            if debug:
                plt.imshow(
                    self.base_dataset_class.mask2color(preds_seg_it[0].argmax(dim=0).cpu().numpy())
                )
                plt.title(f'SEG: {i}')
                plt.show()
            preds_seg.append(preds_seg_it.detach().cpu())

            pred_rec, rec_mse = None, None
            for tta_method, weight in zip(self.tta_methods, self.weights):
                if tta_method == 'iou':
                    loss_seg = self.get_iou_loss(preds_seg_it)
                    losses += [loss_seg.mean()]
                    loss_dict[tta_method].append(loss_seg.squeeze().cpu().detach().numpy())
                if tta_method == 'ref':
                    loss_seg, pm = self.get_refinement_loss(preds_seg_it_raw, loss_name=self.cfg.loss_name)
                    losses += [loss_seg.mean()]
                    loss_dict[tta_method].append(loss_seg.cpu().detach().numpy())
                    if ret_pseudo_masks:
                        pseudo_masks.append(pm)
                if tta_method == 'tent':
                    loss_seg = self.get_tent_loss(preds_seg_it)
                    losses += [loss_seg.mean()]
                    # aggregate over spatial dimension, keep batch dim
                    loss_dict[tta_method].append(loss_seg.mean(axis=(1, 2)).cpu().detach().numpy())
                if tta_method == 'pl':
                    loss_seg, pm = self.get_pseudolabelling_loss(preds_seg_it_raw, loss_name=self.cfg.loss_name)
                    losses += [loss_seg.mean()]
                    # aggregate over spatial dimension, keep batch dim
                    loss_dict[tta_method].append(loss_seg.cpu().detach().numpy())
                    if ret_pseudo_masks:
                        pseudo_masks.append(pm)
                if tta_method == 'augco':
                    loss_seg = self.get_augco_loss(preds_seg_it_raw.detach(), imgs, loss_name=self.cfg.loss_name)
                    losses += [loss_seg.mean()]
                    # aggregate over spatial dimension, keep batch dim
                    loss_dict[tta_method].append(loss_seg.cpu().detach().numpy())
                if tta_method == 'adv':
                    loss_seg = self.get_adversarial_loss(imgs)
                    # note: not mathematically correct KL, TODO move this inside so that we can always compute mean?
                    losses += [loss_seg.mean()]
                    loss_dict[tta_method].append(loss_seg.mean(axis=(1, 2, 3)).cpu().detach().numpy())

            loss = sum([loss * weight for loss, weight in zip(losses, self.weights)])

            if i != num_it:
                # backward on loss
                optim.zero_grad()
                loss.backward()
                # do gradient clipping
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.segmentation_model.parameters(), self.cfg.grad_clip)
                optim.step()

                # free memory
                preds_seg_it = None  # helps to decrease gpu memory used
                gc.collect()
                torch.cuda.empty_cache()

            if debug:
                print_str = f'losses iter {i}: ' + '; '.join(
                    [f'{tta_method}: {val}' for tta_method, val in zip(self.tta_methods, losses)])
                if gt is not None:
                    iou = 1 - ((preds_seg * gt).sum() / ((preds_seg + gt)).sum())
                    print_str += f'; iou: {iou}'
                print(print_str)

        # it x bs x c x h x w, placeholder if no reconstruction TTA is used
        preds_rec = torch.stack(preds_rec) if pred_rec is not None \
            else None
        # insert class dimension
        pseudo_masks = torch.stack(pseudo_masks)[:, :, None] if len(pseudo_masks) > 0 else None
        preds_seg = torch.stack(preds_seg)
        # it x bs x c x h x w - > bs x it x c x h x w
        preds_rec = preds_rec.permute(1, 0, 2, 3, 4) if preds_rec is not None else None
        preds_seg = preds_seg.permute(1, 0, 2, 3, 4)
        pseudo_masks = pseudo_masks.permute(1, 0, 2, 3, 4) if pseudo_masks is not None else None

        # TODO transpose pseudomasks
        return preds_seg, preds_rec, loss_dict, pseudo_masks

    def verify_method(self, method):
        sub_methods = method.split('&')
        # make sure gc is not combined with other methods, as it is not supported yet
        if 'gc' in sub_methods and len(sub_methods) > 1:
            raise ValueError(f'Invalid tta method {method}, gc cannot be combined with other methods')
        for sub in sub_methods:
            if sub not in self.valid_methods:
                raise ValueError(f'Invalid tta method {sub}')
        # sort them
        sub_methods = sorted(sub_methods)
        return sub_methods

    def get_refinement_loss(self, mask_pred_tensor_raw, vis=False, loss_name='ce'):
        mask_ref = self.ref_model(mask_pred_tensor_raw.detach().softmax(dim=1), inference=True)
        if vis:
            plt.subplot(1, 2, 1)
            plt.imshow(mask_ref.squeeze().detach().cpu().numpy())
            plt.title('mask_ref')
            plt.subplot(1, 2, 2)
            plt.imshow(mask_pred_tensor_raw.squeeze().detach().cpu().numpy())
            plt.title('mask_pred')
            plt.show()
        gt_ce = mask_ref.argmax(dim=1)
        if loss_name == 'iou':
            loss = iou_loss(mask_pred_tensor_raw, mask_ref, apply_softmax=True, reduction='none')
        elif loss_name == 'ce':
            loss = torch.nn.functional.cross_entropy(mask_pred_tensor_raw, gt_ce, reduction='none')
            loss = loss.mean(dim=(1, 2))
        return loss, gt_ce.detach().cpu()

    def get_iou_loss(self, mask_pred_tensor):
        loss_pred = self.loss_model(mask_pred_tensor)
        return loss_pred

    def get_tent_loss(self, mask_pred_tensor):
        """
        mask_pred_tensor: tensor of shape (bs, c, h, w), softmax already applied
        """
        output_entropy = torch.sum(
            torch.special.entr(mask_pred_tensor), 1)
        return output_entropy

    def get_adversarial_loss(self, ims_tensor):
        # freeze the full model, we only want to have gradients for the input image
        self.segmentation_model.freeze()

        # compute adversarial images
        # get gt kind with 50 prob of each
        adv_ims_tensor = self.segmentation_model.fgsm_attack(ims_tensor,
                                                             debug=False,
                                                             norm_fun=self.base_dataset_class.normalize,
                                                             inv_norm_fun=self.base_dataset_class.denormalize)

        # reset grad requirements
        self.reset_require_grad()

        # compute KL loss between predictions on adversarial images and original images
        with torch.no_grad():
            # we don't want gradients here
            preds_seg = self.segmentation_model(ims_tensor, inference=False)
        adv_preds_seg = self.segmentation_model(adv_ims_tensor, inference=False)

        # compute reverseKL loss
        loss = torch.nn.functional.kl_div(
            adv_preds_seg,
            preds_seg,
            log_target=True,
            reduction='none'
        )
        return loss

    def get_pseudolabelling_loss(self, mask_pred_tensor_raw, loss_name='iou'):
        gt_mask_ce = mask_pred_tensor_raw.detach().argmax(dim=1, keepdim=True)
        # create a mask with the same size as the prediction, binary according to gt_mask_ce
        gt_mask_iou = torch.zeros_like(mask_pred_tensor_raw)
        gt_mask_iou.scatter_(1, gt_mask_ce, 1)
        if loss_name == 'iou':
            loss = iou_loss(mask_pred_tensor_raw, gt_mask_iou, apply_softmax=True, reduction='none')
        elif loss_name == 'ce':
            loss = torch.nn.functional.cross_entropy(mask_pred_tensor_raw, gt_mask_ce.squeeze(1), reduction='none')
            loss = loss.mean(dim=(1, 2))
        return loss, gt_mask_ce.detach().cpu()

    def get_augco_loss(self, orig_mask_pred_tensor_raw, imgs, debug=False, loss_name='iou'):
        """
        Computes two masks
        M1  - crop the prediction on the original image, resize to original size
        M2  - apply random jitter, crop the bbox in the image, resize to original size, run prediction
        :param orig_mask_pred_tensor_raw:
        :param imgs:
        :return:
        """

        def get_crop_params(orig_h, orig_w):
            """
            Gives cx, cy, h, w of a bounding box with h/w aspect ratio that takes
            25-50% of the image
            :return: top, left, h, w
            """
            orig_area = orig_h * orig_w
            orig_ratio = orig_w / orig_h
            params = RandomResizedCrop.get_params(imgs[0], scale=(0.25, 0.5), ratio=(orig_ratio, orig_ratio))
            bad_params = np.abs((params[3] / params[
                2]) - orig_ratio) > 0.01
            if  bad_params:
                print(f'Aspect ratio difference too big: New - {params[3] / params[2]}; orig - {orig_ratio}, diff: {params[3] / params[2] - orig_ratio}')
                # fallback to using the whole picture
                params = (0, 0, orig_h, orig_w)
            return params

        def get_train_mask(cropped_preds_raw, preds_crop_raw, thresh=0.8, debug=False):
            """
            Returns a mask of pixels that are consistent between the cropped prediction and the prediction on the
            cropped image
            :param cropped_preds_raw: B x C x H x W
            :param preds_crop_raw:  B x C x H x W
            :param thresh: threshold for the confidence of the prediction
            :param debug: if true, visualize the masks
            :return: B x H x W
            """
            pred_crop_argmax = preds_crop_raw.argmax(dim=1)
            consistent_mask = cropped_preds_raw.argmax(dim=1) == pred_crop_argmax
            #  get confidence scores for argmax predictions
            confs = torch.gather(preds_crop_raw.softmax(dim=1), 1, pred_crop_argmax.unsqueeze(1)).squeeze(1)
            confident_mask = confs > thresh
            if debug:
                #     visualize all the masks
                fs = 30
                #  set figsize
                plt.figure(figsize=(20, 8))
                plt.subplot(2, 3, 1)
                plt.imshow(
                    self.base_dataset_class.mask2color(cropped_preds_raw[0].argmax(dim=0).detach().cpu().numpy()))
                plt.title('cropped prediction', fontsize=fs)
                plt.subplot(2, 3, 2)
                plt.imshow(self.base_dataset_class.mask2color(pred_crop_argmax[0].detach().cpu().numpy()))
                plt.title('prediction on cropped', fontsize=fs)
                plt.subplot(2, 3, 3)
                plt.imshow(consistent_mask[0].detach().cpu().numpy())
                plt.title('consistent mask', fontsize=fs)
                plt.subplot(2, 3, 4)
                plt.imshow(confident_mask[0].detach().cpu().numpy())
                plt.title('confident mask', fontsize=fs)
                plt.subplot(2, 3, 5)
                plt.imshow((consistent_mask | confident_mask)[0].detach().cpu().numpy())
                plt.title('consistent | confident', fontsize=fs)

                # set all axis off
                for i in range(6):
                    plt.subplot(2, 3, i + 1)
                    plt.axis('off')
                plt.tight_layout()
                plt.show()

            return consistent_mask | confident_mask

        # set anomaly detection
        torch.autograd.set_detect_anomaly(True)

        crop_bbox = get_crop_params(imgs.shape[-2], imgs.shape[-1])

        if self.base_dataset == "coco":
            imgs = imgs.permute(0, 2, 3, 1)
        denorm_imgs = self.base_dataset_class.denormalize(imgs).to(torch.uint8)
        if self.base_dataset == "coco":
            denorm_imgs = denorm_imgs.permute(0, 3, 1, 2)
            imgs = imgs.permute(0, 3, 1, 2)
        # also move to rgb
        denorm_imgs = denorm_imgs[:, [2, 1, 0], :, :]
        #   apply random jitter
        imgs_jit = self.color_augment(denorm_imgs)

        if debug:
            plt.imshow(denorm_imgs[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.show()
            plt.imshow(imgs_jit[0].permute(1, 2, 0).detach().cpu().numpy())
            plt.show()

        #  normalize again
        if self.base_dataset == "coco":
            imgs_jit = imgs_jit.cpu().detach().numpy().squeeze()
            imgs_jit = np.transpose(imgs_jit, (1, 2, 0))

        imgs_jit = self.base_dataset_class.normalize(imgs_jit)

        if self.base_dataset == "coco":
            imgs_jit = imgs_jit.permute(2, 0, 1).unsqueeze(0).to(imgs.device)

        # move back to bgr
        imgs_jit = imgs_jit[:, [2, 1, 0], :, :]
        # crop the image
        imgs_cropped = TF.resized_crop(imgs_jit, crop_bbox[0], crop_bbox[1], crop_bbox[2], crop_bbox[3],
                                       size=(imgs.shape[-2], imgs.shape[-1]))
        #   resize to original image size
        imgs_cropped = F.interpolate(imgs_cropped, size=(imgs.shape[-2], imgs.shape[-1]), mode='bilinear',
                                     align_corners=False)
        #   run prediction
        preds_on_cropped_raw = self.segmentation_model(imgs_cropped, inference=False)

        #  crop orig preditcion
        cropped_orig_preds_raw = TF.resized_crop(orig_mask_pred_tensor_raw, crop_bbox[0], crop_bbox[1], crop_bbox[2],
                                                 crop_bbox[3], size=(imgs.shape[-2], imgs.shape[-1]))

        train_mask = get_train_mask(preds_on_cropped_raw, cropped_orig_preds_raw, debug=debug)[:, None]

        # compute pseudo-label loss from prediction on cropped image, using the train mask
        gt_ce = preds_on_cropped_raw.detach().argmax(dim=1, keepdim=True)
        if loss_name == 'ce':
            gt_ce[~train_mask] = 255
            loss = torch.nn.functional.cross_entropy(preds_on_cropped_raw, gt_ce.squeeze(1), reduction='none',
                                                     ignore_index=255)
            loss = loss.mean(axis=(1, 2))
        elif loss_name == 'iou':
            # create a mask with the same size as the prediction, binary according to gt_mask_ce
            gt_iou = torch.zeros_like(preds_on_cropped_raw)
            gt_iou.scatter_(1, gt_ce, 1)

            # set gt_iou to 255 where train_mask is false across all channels
            gt_iou[~train_mask.repeat(1, gt_iou.shape[1], 1, 1)] = 255

            loss = iou_loss(preds_on_cropped_raw, gt_iou, reduction='none', apply_softmax=True, ignore_index=255)
        return loss

    def forward_segmentation(self, im_tensors, inference=True):
        # if 'rec' not in self.tta_methods:
        #     raise ValueError('Reconstruction TTA method must be used to be abel to forward segmentation')
        if self.segmentation_model is None:
            # lazy segmentation model loading
            self.segmentation_model = get_seg_model(
                size=(im_tensors.shape[-1], im_tensors.shape[-2]),
                path=os.path.join(self.ckpt_dir, self.model_path) if self.model_path is not None else None,
            )
            self.segmentation_model.to(device)
            self.segmentation_model.eval()
        with torch.no_grad():
            preds_seg = self.segmentation_model(im_tensors, inference=inference)
        return preds_seg

    def forward_refinement(self, masks_pred_tensor, vis=True):
        masks_ref = self.ref_net(masks_pred_tensor)
        if vis:
            plt.subplot(1, 2, 1)
            plt.imshow(masks_ref[0].squeeze().detach().cpu().numpy())
            plt.title('mask_ref')
            plt.subplot(1, 2, 2)
            plt.imshow(masks_pred_tensor[0].squeeze().detach().cpu().numpy())
            plt.title('mask_pred')
            plt.show()
        return masks_ref


class SubRandAugment(RandAugment):
    """
    Subclass of RandAugment that allows to specify a subset of transformations to use
    """

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[F.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
