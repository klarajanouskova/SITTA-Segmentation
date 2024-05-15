import os
from datetime import datetime

import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import numpy as np
from model.deeplab import Res_Deeplab
from data.driving_dataset import DrivingDataset
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class Segmenter(torch.nn.Module):
    def pgd_attack(self, ims_tensor, norm_fun, inv_norm_fun, iters=40, lr=0.005, thresh=0.4, epsilon=0.05,
                   debug=False, gt='invert'):
        raise NotImplementedError

    def fgsm_attack(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def pred2colormask(self, pred):
        pred = pred.argmax(dim=0, keepdim=True)
        mask = DrivingDataset.mask2color(pred.squeeze().cpu().numpy())
        return mask


class DeepLabSegmenter(Segmenter):
    def __init__(self, size, num_classes=19, path='ckpts/gta5/dr2_model.pth'):
        super().__init__()
        self.model = Res_Deeplab(num_classes=num_classes)
        saved_state_dict = torch.load(path)
        self.model.load_state_dict(saved_state_dict, strict=True)
        self.upsample = torch.nn.Upsample(size=(size[1], size[0]), mode='bilinear', align_corners=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, inference=True, upsample=True):
        """
        :param x:
        :param inference: if True, apply softmax to logits  (during training, it is often part of the loss)
        :param upsample: if True, upsample logits to the original image size
        :return:
        """
        logits = self.model(x)
        if upsample:
            logits = self.upsample(logits)
        if inference:
            return torch.softmax(logits, dim=1)
        else:
            return logits

    def train_norm_layers_only(self):
        """
                Freeze all layers except for the normalization layers - layernorm in transformer, batchnorm in convnets.
                Usefull for example for the TENT TTA method.
                """

        for name, p in self.named_parameters():
            if 'norm' in name.lower() or '.bn' in name.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def pgd_attack(self, ims_tensor, norm_fun, inv_norm_fun, iters=40, lr=0.5, thresh=0.4, epsilon=1,
                   debug=False, gt='invert', vis_dir=None):
        """
           Projected gradient descent adversarial attack, L_inf norm clipping
           Assumes segmentation model_seg has frozen weights
           """

        assert gt in ['invert', 'random']
        if vis_dir:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            vis_dir = os.path.join(vis_dir, 'adv', timestamp)
            os.makedirs(vis_dir, exist_ok=True)

        ims_adv_tensor = ims_tensor.clone().detach()
        ims_adv_tensor.requires_grad = True
        optim = torch.optim.Adam([ims_adv_tensor], lr=lr)

        for it in range(iters):
            # Forward pass the base_data through the model_seg
            seg_pred = self(ims_adv_tensor)

            if it == 0:
                if gt == 'random':
                    # generate random gt
                    r = (torch.rand_like(seg_pred) > thresh).float()
                    # blur gt
                    r = torch.nn.functional.avg_pool2d(r, 5, stride=1, padding=2)
                    gt = seg_pred.clone()
                    # gt[r > 0] = r[r > 0].detach()
                    gt = torch.clamp(r + (r * gt) * 0.5, 0, 1)

                elif gt == 'invert':
                    gt = (seg_pred < thresh).float()
                #     otherwise backward fails since we pass it multiple times
                gt = gt.detach()

                seg_preds_orig = seg_pred.clone()

            # Calculate the loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(seg_pred, gt)

            # Zero all existing gradients
            optim.zero_grad()

            # Calculate gradients of model_seg in backward pass
            loss.backward()

            # perform optimization step
            optim.step()

            # clipping to max epsilon difference
            delta = ims_adv_tensor.detach() - ims_tensor
            delta_norm = torch.abs(delta)
            div = torch.clamp(delta_norm / epsilon, min=1.0)
            delta = delta / div
            ims_adv_tensor.data = ims_tensor + delta

            # visualize input image, adversarial image, their difference, and the segmentation masks
            # for the first 5 classes
            if debug and (it % 1 == 0 or it == iters - 1):
                perturbed_images = ims_adv_tensor.clone()
                perturbed_images.detach()
                seg_pred_pert = self(perturbed_images)
                for img, img_adv, seg_orig, seg_adv in zip(ims_tensor, perturbed_images, seg_preds_orig, seg_pred_pert):
                    # first show gt
                    if it == 0:
                        vis_gt = torch.hstack([gt[0, i, :, :] for i in range(5)])
                        plt.imshow(vis_gt.squeeze().detach().cpu().numpy())
                        # plt.set_cmap('gray')
                        plt.title('gt')
                        # no white space around plot
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                        img_vis = to_pil_image(inv_norm_fun(img.cpu())[[2, 1, 0]].to(torch.uint8))
                        plt.imshow(img_vis)
                        if vis_dir is not None:
                            img_vis.save(os.path.join(vis_dir, f'img.png'))
                        plt.title('input')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                        seg_orig_vis = self.pred2colormask(seg_orig)
                        plt.imshow(seg_orig_vis)
                        plt.title('segmentation')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                        if vis_dir is not None:
                            # convert to rgb
                            seg_orig_vis = seg_orig_vis.convert('RGB')
                            seg_orig_vis.save(os.path.join(vis_dir, f'seg_0.png'))


                    # plt.imshow(to_pil_image(inv_norm_fun(img_adv.cpu())[[2, 1, 0]].to(torch.uint8)))
                    # plt.title(f'adversarial {it}')
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.show()
                    # plt.imshow(to_pil_image(inv_norm_fun((img - img_adv).cpu())[[2, 1, 0]].to(torch.uint8)))
                    # plt.title('diff')
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.show()
                    # visualize the predictions for first 5 classes
                    seg_adv_vis = self.pred2colormask(seg_adv)
                    # plt seg orig and seg adv next to each other as subplots
                    plt.subplot(1, 2, 1)
                    plt.imshow(seg_orig_vis)
                    plt.title('seg orig')
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(seg_adv_vis)
                    plt.title(f'seg adv {it}')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                    pixels_changed = torch.sum((seg_orig > thresh) != (seg_adv > thresh))
                    print(f'pixels changed in {it}: {pixels_changed}, lr: {lr}, eps: {epsilon}')
                    # plt.suptitle(
                    #     f'Adversarial perturbation, eps: {epsilon}; lr: {lr} # iter: {it}, pixels changed: {pixels_changed}')
                    if vis_dir is not None:
                        # convert to rgb
                        seg_adv_vis = seg_adv_vis.convert('RGB')
                        seg_adv_vis.save(os.path.join(vis_dir, f'seg_{it + 1}.png'))
                    # hide all axes
                    plt.show()
                    print()
        # denormalize, clip the output, renormalize
        ims_adv_tensor = ims_adv_tensor.detach()
        perturbed_images = inv_norm_fun(ims_adv_tensor)
        perturbed_images = torch.clamp(perturbed_images, 0, 255)
        perturbed_images = norm_fun(perturbed_images)

        # Return the perturbed image
        return perturbed_images, timestamp

    def fgsm_attack(self, ims_tensor, norm_fun, inv_norm_fun, thresh=0.4, epsilon=1, debug=False, gt='invert'):
        """
        FGSM adversarial attack
        """

        assert gt in ['invert', 'random']

        # set model_seg requires grad to false
        self.freeze()

        ims_adv_tensor = ims_tensor.clone().detach()
        ims_adv_tensor.requires_grad = True

        # Forward pass the base_data through the model_seg
        seg_pred = self(ims_adv_tensor, inference=True)

        # generate random gt
        if gt == 'random':
            # generate random gt
            r = (torch.rand_like(seg_pred) > thresh).float()
            # blur gt
            r = torch.nn.functional.avg_pool2d(r, 5, stride=1, padding=2)
            gt = seg_pred.clone()
            # gt[r > 0] = r[r > 0].detach()
            gt = torch.clamp(r + (r * gt) * 0.5, 0, 1)

        elif gt == 'invert':
            gt = (seg_pred < thresh).float()

        # Calculate the loss
        # TODO test this without sigmpid applies
        loss = torch.nn.functional.binary_cross_entropy_with_logits(seg_pred, gt)

        # zero image gradients
        ims_adv_tensor.grad = None

        # Calculate gradients of model_seg in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = ims_adv_tensor.grad.data
        # Collect the element-wise sign of the base_data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_images = ims_adv_tensor + epsilon * sign_data_grad

        # no clipping here  since we have standardized images??

        # visualize input image, adversarial image, their difference, and the segmentation masks
        # if debug:
        #     seg_pred_pert = segmentation_model.forward_seg(perturbed_images, inference=True)
        #     for img, img_adv, seg, seg_adv in zip(ims_tensor, perturbed_images, seg_pred, seg_pred_pert):
        #         # run segmentation on the perturbed image
        #         plt.subplots(2, 3, figsize=(10, 6))
        #         plt.subplot(2, 3, 1)
        #         plt.imshow(to_pil_image(inv_norm_fun(img.cpu())))
        #         plt.title('input')
        #         plt.subplot(2, 3, 2)
        #         plt.imshow(to_pil_image(inv_norm_fun(img_adv.cpu())))
        #         plt.title('adversarial')
        #         plt.subplot(2, 3, 3)
        #         plt.imshow(to_pil_image(inv_norm_fun((img - img_adv).cpu())))
        #         plt.title('diff')
        #         plt.subplot(2, 3, 4)
        #         plt.imshow(seg.squeeze().detach().cpu().numpy())
        #         plt.title('seg orig')
        #         plt.subplot(2, 3, 5)
        #         plt.imshow(seg_adv.squeeze().detach().cpu().numpy())
        #         plt.title('seg adv')
        #         pixels_changed = torch.sum((seg > thresh) != (seg_adv > thresh))
        #         plt.suptitle(
        #             f'Adversarial perturbation, eps: {epsilon}; pixels changed: {pixels_changed}')
        #         # hide all axes
        #         for ax in plt.gcf().axes:
        #             ax.axis('off')
        #         # increase title fonts
        #         for ax in plt.gcf().axes:
        #             ax.title.set_fontsize(15)
        #         plt.show()

        # denormalize, clip the output, renormalize
        perturbed_images = inv_norm_fun(perturbed_images.detach())
        perturbed_images = torch.clamp(perturbed_images, 0, 255)
        perturbed_images = norm_fun(perturbed_images)

        # Return the perturbed image
        return perturbed_images


class COCODeepLabSegmenter(Segmenter):
    def __init__(self, size):
        super().__init__()
        self.model = deeplabv3_resnet50(DeepLabV3_ResNet50_Weights)
        #self.model.load_state_dict(
        #    torch.load("/home/tamirshor/tta_rec/pretrained_weights/DeepLab/deeplabv3_resnet50_coco-cd0a2569.pth"))
        if isinstance(size,int):
            size = (size,size)
        self.upsample = torch.nn.Upsample(size=(size[1], size[0]), mode='bilinear', align_corners=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, inference=True, upsample=True):
        """
        :param x:
        :param inference: if True, apply softmax to logits  (during training, it is often part of the loss)
        :param upsample: if True, upsample logits to the original image size
        :return:
        """

        logits = self.model(x)['out']
        if upsample:
            logits = self.upsample(logits)
        if inference:
            return torch.softmax(logits, dim=1)
        else:
            return logits


    def train_norm_layers_only(self):
        """
                Freeze all layers except for the normalization layers - layernorm in transformer, batchnorm in convnets.
                Usefull for example for the TENT TTA method.
                """

        for name, p in self.named_parameters():
            if 'norm' in name.lower() or '.bn' in name.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def pgd_attack(self, ims_tensor, norm_fun, inv_norm_fun, iters=40, lr=0.5, thresh=0.4, epsilon=1,
                   debug=False, gt='invert'):
        """
           Projected gradient descent adversarial attack, L_inf norm clipping
           Assumes segmentation model_seg has frozen weights
           """

        assert gt in ['invert', 'random']

        ims_adv_tensor = ims_tensor.clone().detach()
        ims_adv_tensor.requires_grad = True
        optim = torch.optim.Adam([ims_adv_tensor], lr=lr)
        device = ims_adv_tensor.device


        for it in range(iters):
            # Forward pass the base_data through the model_seg
            seg_pred = self(ims_adv_tensor)

            if it == 0:
                if gt == 'random':
                    # generate random gt
                    r = (torch.rand_like(seg_pred) > thresh).float()
                    # blur gt
                    r = torch.nn.functional.avg_pool2d(r, 5, stride=1, padding=2)
                    gt = seg_pred.clone()
                    # gt[r > 0] = r[r > 0].detach()
                    gt = torch.clamp(r + (r * gt) * 0.5, 0, 1)

                elif gt == 'invert':
                    gt = (seg_pred < thresh).float()
                #     otherwise backward fails since we pass it multiple times
                gt = gt.detach()

                seg_preds_orig = seg_pred.clone()

            # Calculate the loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(seg_pred, gt)

            # Zero all existing gradients
            optim.zero_grad()

            # Calculate gradients of model_seg in backward pass
            loss.backward()

            # perform optimization step
            optim.step()

            # clipping to max epsilon difference
            delta = ims_adv_tensor.detach() - ims_tensor
            delta_norm = torch.abs(delta)
            div = torch.clamp(delta_norm / epsilon, min=1.0)
            delta = delta / div
            ims_adv_tensor.data = ims_tensor + delta

            # visualize input image, adversarial image, their difference, and the segmentation masks
            # for the first 5 classes
            if debug and (it % 1 == 0 or it == iters - 1):
                perturbed_images = ims_adv_tensor.clone()
                perturbed_images.detach()
                seg_pred_pert = self(perturbed_images)
                for img, img_adv, seg_orig, seg_adv in zip(ims_tensor, perturbed_images, seg_preds_orig, seg_pred_pert):
                    # first show gt
                    if it == 0:
                        vis_gt = torch.hstack([gt[0, i, :, :] for i in range(5)])
                        plt.imshow(vis_gt.squeeze().detach().cpu().numpy())
                        plt.set_cmap('gray')
                        plt.title('gt')
                        # no white space around plot
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                        plt.imshow(to_pil_image((inv_norm_fun(img.cpu().permute(1,2,0)).permute(2,0,1))[[2, 1, 0]].to(torch.uint8)))
                        plt.title('input')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show()
                        seg_orig_vis = self.pred2colormask(seg_orig)
                    # plt.imshow(to_pil_image(inv_norm_fun(img_adv.cpu())[[2, 1, 0]].to(torch.uint8)))
                    # plt.title(f'adversarial {it}')
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.show()
                    # plt.imshow(to_pil_image(inv_norm_fun((img - img_adv).cpu())[[2, 1, 0]].to(torch.uint8)))
                    # plt.title('diff')
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.show()
                    # visualize the predictions for first 5 classes
                    seg_adv_vis = self.pred2colormask(seg_adv)
                    # plt seg orig and seg adv next to each other as subplots
                    plt.subplot(1, 2, 1)
                    plt.imshow(seg_orig_vis)
                    plt.title('seg orig')
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(seg_adv_vis)
                    plt.title(f'seg adv {it}')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                    pixels_changed = torch.sum((seg_orig > thresh) != (seg_adv > thresh))
                    print(f'pixels changed in {it}: {pixels_changed}, lr: {lr}, eps: {epsilon}')
                    # plt.suptitle(
                    #     f'Adversarial perturbation, eps: {epsilon}; lr: {lr} # iter: {it}, pixels changed: {pixels_changed}')
                    # hide all axes
                    plt.show()
                    print()
        # denormalize, clip the output, renormalize
        ims_adv_tensor = ims_adv_tensor.detach()
        perturbed_images = inv_norm_fun(ims_adv_tensor.permute(0,2,3,1))
        perturbed_images = torch.clamp(perturbed_images, 0, 255)
        perturbed_images = norm_fun(perturbed_images[0].cpu().detach().numpy().astype(np.uint8)).unsqueeze(0)

        # Return the perturbed image
        return perturbed_images.to(device)

    def fgsm_attack(self, ims_tensor, norm_fun, inv_norm_fun, thresh=0.4, epsilon=0.1, debug=False, gt='invert'):
        """
        FGSM adversarial attack
        """

        assert gt in ['invert', 'random']

        # set model_seg requires grad to false
        self.freeze()

        ims_adv_tensor = ims_tensor.clone().detach()
        ims_adv_tensor.requires_grad = True

        # Forward pass the base_data through the model_seg
        seg_pred = self(ims_adv_tensor, inference=True)

        # generate random gt
        if gt == 'random':
            # generate random gt
            r = (torch.rand_like(seg_pred) > thresh).float()
            # blur gt
            r = torch.nn.functional.avg_pool2d(r, 5, stride=1, padding=2)
            gt = seg_pred.clone()
            # gt[r > 0] = r[r > 0].detach()
            gt = torch.clamp(r + (r * gt) * 0.5, 0, 1)

        elif gt == 'invert':
            gt = (seg_pred < thresh).float()

        # Calculate the loss
        # TODO test this without sigmpid applies
        loss = torch.nn.functional.binary_cross_entropy_with_logits(seg_pred, gt)

        # zero image gradients
        ims_adv_tensor.grad = None

        # Calculate gradients of model_seg in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = ims_adv_tensor.grad.data
        # Collect the element-wise sign of the base_data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_images = ims_adv_tensor + epsilon * sign_data_grad



        # denormalize, clip the output, renormalize
        perturbed_images = inv_norm_fun(perturbed_images.detach().permute(0,2,3,1))
        perturbed_images = torch.clamp(perturbed_images, 0, 255)
        perturbed_images = norm_fun(perturbed_images[0].cpu().detach().numpy().astype(np.uint8)).unsqueeze(0)

        # Return the perturbed image
        return perturbed_images.permute(0,3,1,2).to(ims_tensor.device)
