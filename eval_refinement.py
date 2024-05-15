import torch
import numpy as np
from pathlib import Path
import torchvision
from torchmetrics.classification import MulticlassAccuracy

torchvision.disable_beta_transforms_warning()

import hydra
from omegaconf import OmegaConf
import os

import sys
sys.path.append('')

from model.segmenter import DeepLabSegmenter,COCODeepLabSegmenter

from tqdm import tqdm
from torchmetrics.classification.jaccard import MulticlassJaccardIndex
import matplotlib.pyplot as plt

from data.cityscapes import get_cityscapes
from data.gta5 import get_gta5
from data.acdc import get_acdc, ACDC

from distortion import distortions_cov
from util.misc import load_tta_model
from models_loss import MaskLossUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DATASET = "gta5" #gta5 or coco
num_classes = 21 if BASE_DATASET=='coco' else 19

REF_WEIGHT_LIST = [
    "sem_seg_deep_loss_softmax_gta5_ref_adversarial_use_gt0_lr0.001_adv_lr0.1_seed0_08-04-13-10",
    "sem_seg_deep_loss_softmax_gta5_ref_adversarial_use_gt1_lr0.001_adv_lr0.1_seed0_08-04-22-34",
    "sem_seg_deep_loss_softmax_gta5_ref_corruption_use_gt0_lr0.001_adv_lr0.1_seed0_08-05-07-55",
    "sem_seg_deep_loss_softmax_gta5_ref_corruption_use_gt1_lr0.001_adv_lr0.1_seed0_08-05-13-17"
]


def get_seg_model(size, path='ckpts/gta5/dr2_model.pth'):
    if BASE_DATASET=="coco":
        model = COCODeepLabSegmenter(size=size)
    else:
        model = DeepLabSegmenter(num_classes=num_classes, path=path, size=size)
    return model


def get_dataset(cfg):
    if cfg.name == "gta5":
        dataset = get_gta5(split='val', val_size=250)
    elif cfg.name == "cityscapes":
        dataset = get_cityscapes(split='val', condition='clean')
    elif cfg.name.startswith("acdc"):
        condition = cfg.name.split("-")[1]
        dataset = get_acdc(condition=condition)
    else:
        raise NotImplementedError("Dataset not implemented")
    return dataset


def create_output_dirs(cfg):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    # make ckpt dir in output dir
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    # make logs dir in output dir
    Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    # make results dir in output dir
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)


def complete_config(cfg, print_cfg=False):
    # TODO complete this?
    cfg.exp_name = f"{cfg.exp_name}_{cfg.base_data.name}_seed{cfg.seed}"

    print(f'Running experiment {cfg.exp_name}')
    log_dir = os.path.join(cfg.out_dir, 'logs')
    cfg.log_dir = log_dir
    results_dir = os.path.join(cfg.out_dir, 'results')
    cfg.results_dir = results_dir

    if print_cfg:
        print(OmegaConf.to_yaml(cfg))


@hydra.main(config_path=f"conf/{'coco' if BASE_DATASET=='coco' else 'gta'}", config_name="tta_eval", version_base=None)
def sweep_refinement_cityscapes(cfg):
    """
    Evaluate maks refinement as post-processing
    :param cfg:
    :return:
    """

    # allow adding new keys
    OmegaConf.set_struct(cfg, False)
    complete_config(cfg, print_cfg=True)
    create_output_dirs(cfg)
    cfg.bs = 8

    ckpt_id = 'checkpoint-best.pth'

    cfg.base_data.name = "cityscapes"
    dataset = get_dataset(cfg)

    model_seg = get_seg_model(size=dataset.crop_size)
    model_seg.eval()
    model_seg.to(device)

    save_folder = f"refinement-post/cityscapes"

    # with open(log_file, 'w') as f:
    for weights_id in REF_WEIGHT_LIST:
        save_path = os.path.join(cfg.results_dir, save_folder, weights_id)
        weights = f"{weights_id}/{ckpt_id}"
        ckpt_path = os.path.join(cfg.ckpt_dir, weights)
        model_ref = load_tta_model(ckpt_path, MaskLossUnet, size=dataset.crop_size, num_classes=19)
        model_ref.eval()
        model_ref.to(device)
        # f.write(f"Testing refinement {weights_id}\n")
        print(f"Testing refinement {weights_id}")
        test_clean(cfg, dataset, model_seg, model_ref, n_samples=500, log_file=None, vis=False, save_path=save_path)


@hydra.main(config_path=f"conf/{'coco' if BASE_DATASET=='coco' else 'gta'}", config_name="tta_eval", version_base=None)
def sweep_refinement_acdc(cfg):
    # allow adding new keys
    OmegaConf.set_struct(cfg, False)
    complete_config(cfg, print_cfg=True)
    create_output_dirs(cfg)
    cfg.bs = 8

    ckpt_id = 'checkpoint-best.pth'
    for cond in ACDC.VALID_CONDS:
        save_folder = f"refinement-post/acdc_{cond}"
        # create folder
        Path(os.path.join(cfg.results_dir, save_folder)).mkdir(parents=True, exist_ok=True)
        #    acdc night
        cfg.base_data.name = f"acdc-{cond}"
        dataset = get_dataset(cfg)

        model_seg = get_seg_model(size=dataset.crop_size)
        model_seg.eval()
        model_seg.to(device)

        for weights_id in REF_WEIGHT_LIST:
            save_path = os.path.join(cfg.results_dir, save_folder, weights_id)
            weights = f"{weights_id}/{ckpt_id}"
            ckpt_path = os.path.join(cfg.ckpt_dir, weights)
            model_ref = load_tta_model(ckpt_path, MaskLossUnet, size=dataset.crop_size, num_classes=19)
            model_ref.eval()
            model_ref.to(device)
            print(f"Testing refinement {weights_id}")
            test_clean(cfg, dataset, model_seg, model_ref, n_samples=500, log_file=None, vis=False, save_path=save_path)



@hydra.main(config_path=f"conf/{'coco' if BASE_DATASET=='coco' else 'gta'}", config_name="tta_eval", version_base=None)
def ablate_refinement_corruptions(cfg):
    """
    Abaltion study on different refinement-based models on synthetically corrupted images
    :return:
    """

    # set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # allow adding new keys
    OmegaConf.set_struct(cfg, False)
    complete_config(cfg, print_cfg=True)
    create_output_dirs(cfg)
    cfg.bs = 8

    cfg.base_data.name = "gta5"
    dataset = get_dataset(cfg)

    model_seg = get_seg_model(size=dataset.crop_size)
    model_seg.eval()
    model_seg.to(device)
    ckpt_id = 'checkpoint-best.pth'

    save_folder = f"refinement-post/gta5-c.txt"

    for weights_id in REF_WEIGHT_LIST:
        save_path = os.path.join(cfg.results_dir, save_folder, weights_id)
        weights = f"{weights_id}/{ckpt_id}"
        ckpt_path = os.path.join(cfg.ckpt_dir, weights)
        model_ref = load_tta_model(ckpt_path, MaskLossUnet, size=dataset.crop_size, num_classes=19)
        model_ref.eval()
        model_ref.to(device)
        print(f"Testing {weights_id}")
        test_corruptions(cfg, dataset, model_seg, model_ref, n_samples=50, save_path=save_path)


def test_clean(cfg, dataset, model_seg, model_ref, n_samples=500, log_file=None, vis=False, save_path=None):
    if n_samples < 0:
        n_samples = len(dataset)

    print(50 * "=")

    baseline_iou = MulticlassJaccardIndex(num_classes=19, average='none', ignore_index=255).to(device)
    baseline_acc = MulticlassAccuracy(num_classes=19, average='macro', ignore_index=255).to(device)
    refined_iou = MulticlassJaccardIndex(num_classes=19, average='none', ignore_index=255).to(device)
    refined_acc = MulticlassAccuracy(num_classes=19, average='macro', ignore_index=255).to(device)

    # make dataset subset of n_samples
    subdataset = torch.utils.data.Subset(dataset, range(n_samples))
    # make a dataloader
    dataloader = torch.utils.data.DataLoader(subdataset, batch_size=cfg.bs, shuffle=False, num_workers=0 if vis else 4)

    glob_in_clses_ref,  glob_in_clses_base = torch.zeros(19).cuda(), torch.zeros(19).cuda()

    for i, (img, target, _, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = img.to(device)
        target = target.to(device)

        with torch.no_grad():
            pred_base = model_seg(img, inference=True)
            pred_ref = model_ref(pred_base)

        # baseline
        iou_im_base = baseline_iou(pred_base, target)
        in_clses = torch.hstack([torch.unique(target), torch.unique(pred_base.argmax(dim=0))]).unique()
        glob_in_clses_base[torch.isin(torch.arange(19).cuda(), in_clses)] = 1
        iou_im_base[~torch.isin(torch.arange(19).cuda(), in_clses)] = np.nan
        acc_im_base = baseline_acc(pred_base, target)

        # refined
        iou_im_ref = refined_iou(pred_ref, target)
        in_clses = torch.hstack([torch.unique(target), torch.unique(pred_ref.argmax(dim=0))]).unique()
        glob_in_clses_ref[torch.isin(torch.arange(19).cuda(), in_clses)] = 1
        iou_im_ref[~torch.isin(torch.arange(19).cuda(), in_clses)] = np.nan
        acc_im_ref = refined_acc(pred_ref, target)

        miou_im_base = iou_im_base.nanmean() * 100
        miou_im_ref = iou_im_ref.nanmean() * 100


        # visualize im, gt, pred_base, pred_ref
        if vis:
            plt.figure(figsize=(20, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(dataset.tensor2pil(img[0]))
            plt.title("Image", fontsize=20)
            plt.subplot(2, 2, 2)
            plt.imshow(dataset.mask2color(target[0].cpu().numpy()))
            plt.title("GT", fontsize=20)
            plt.subplot(2, 2, 3)
            plt.imshow(dataset.mask2color(pred_base[0].argmax(dim=0).cpu().numpy()))
            plt.title("Baseline", fontsize=20)
            plt.subplot(2, 2, 4)
            plt.imshow(dataset.mask2color(pred_ref[0].argmax(dim=0).cpu().numpy()))
            plt.title("Refined", fontsize=20)
            # set all axes off
            for ax in plt.gcf().get_axes():
                ax.set_axis_off()
            plt.show()

            # print(f"Image {i} Acc: Base - {acc_im_base}, refined - {acc_im_ref}, improvement - {acc_im_ref - acc_im_base}")
            # print(f"Image {i} IoU: Base - {miou_im_base}, refined - {miou_im_ref}, improvement - {miou_im_ref - miou_im_base}")

            # print per-class results
            # for i in range(19):
            #     print(
            #         f"Class {dataset.CLASS2CAT[i]}-{i}: Base - {iou_im_base[i]}, refined - {iou_im_ref[i]}, "
            #         f"improvement - {iou_im_ref[i] - iou_im_base[i]}")

    class_iou_base = baseline_iou.compute()
    class_iou_ref = refined_iou.compute()

    # set mious of classes that were never in the predictions/gts to nan
    class_iou_base[~glob_in_clses_base.bool()] = np.nan
    class_iou_ref[~glob_in_clses_ref.bool()] = np.nan

    # per class
    miou_base = class_iou_base.nanmean() * 100
    miou_ref = class_iou_ref.nanmean() * 100

    acc_base, acc_ref = baseline_acc.compute() * 100, refined_acc.compute() * 100

    # # print per-class results
    # for i in range(19):
    #     print(
    #         f"Class {dataset.CLASS2CAT[i]}-{i}: Base - {class_iou_base[i]}, refined - {class_iou_ref[i]}, "
    #         f"improvement - {class_iou_ref[i] - class_iou_base[i]}")

    print(f"Baseline mIoU: {miou_base}")
    print(f"Refined mIoU: {miou_ref}")
    print(f"Improvement mIoU: {miou_ref - miou_base}")
    print(f"Baseline Acc: {acc_base}")
    print(f"Refined Acc: {acc_ref}")
    print(f"Improvement Acc: {acc_ref - acc_base}")

    print(50 * "=")

    for i in range(19):
        print(f"Class {dataset.CLASS2CAT[i]}-{i}: Base - {class_iou_base[i]}, refined - {class_iou_ref[i]}")
        if log_file is not None:
            log_file.write(
                f"Class {dataset.CLASS2CAT[i]}-{i}: Base - {class_iou_base[i]}, refined - {class_iou_ref[i]}\n")

    np.savez(save_path, class_iou_base=class_iou_base.cpu().numpy(), class_iou_ref=class_iou_ref.cpu().numpy(),
             miou_base=miou_base.cpu().numpy(), miou_ref=miou_ref.cpu().numpy(), acc_base=acc_base.cpu().numpy(),
                acc_ref=acc_ref.cpu().numpy())


def test_corruptions(cfg, dataset, model_seg, model_ref, distortions=distortions_cov, n_samples=30, vis=False, save_path=None):
    print(50 * "=")
    if n_samples < 0:
        n_samples = len(dataset)

    distortion_keys = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'fog', 'brightness',
        'contrast', 'pixelate', 'jpeg', 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
    ]

    baseline_iou = MulticlassJaccardIndex(num_classes=19, average='none', ignore_index=255).to(device)
    refined_iou = MulticlassJaccardIndex(num_classes=19, average='none', ignore_index=255).to(device)
    bazeline_acc = MulticlassAccuracy(num_classes=19, average='none', ignore_index=255).to(device)
    refined_acc = MulticlassAccuracy(num_classes=19, average='none', ignore_index=255).to(device)

    # make dataset subset of n_samples
    subdataset = torch.utils.data.Subset(dataset, range(n_samples))
    # make a dataloader
    dataloader = torch.utils.data.DataLoader(subdataset, batch_size=cfg.bs, shuffle=False, num_workers=0 if vis else 4)

    glob_in_clses_ref,  glob_in_clses_base = torch.zeros(19), torch.zeros(19)

    for dist_name, dist_fun in distortions.items():
        for severity in range(1, 6):
            print(f"Testing {dist_name} severity {severity}")
            for i, (imgs, target, _, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
                imgs = imgs.to(device)
                target = target.to(device)

                dist_imgs = torch.stack([dataset.pil2tensor(dist_fun(dataset.tensor2pil(im), severity=severity)) for im in imgs], dim=0)
                dist_imgs = dist_imgs.to(device)
                with torch.no_grad():
                    pred_base = model_seg(dist_imgs, inference=True)
                    pred_ref = model_ref(pred_base, inference=True)

                # baseline
                iou_im_base = baseline_iou(pred_base, target)
                in_clses = torch.hstack([torch.unique(target), torch.unique(pred_base.argmax(dim=0))]).unique()
                glob_in_clses_base[torch.isin(torch.arange(19).cuda(), in_clses)] = 1
                iou_im_base[~torch.isin(torch.arange(19).cuda(), in_clses)] = np.nan
                acc_im_base = bazeline_acc(pred_base, target)
                # refined
                iou_im_ref = refined_iou(pred_ref, target)
                in_clses = torch.hstack([torch.unique(target), torch.unique(pred_ref.argmax(dim=0))]).unique()
                glob_in_clses_ref[torch.isin(torch.arange(19).cuda(), in_clses)] = 1
                iou_im_ref[~torch.isin(torch.arange(19).cuda(), in_clses)] = np.nan
                acc_im_ref = refined_acc(pred_ref, target)


    print(50 * "=")

    class_iou_base = baseline_iou.compute()
    class_iou_ref = refined_iou.compute()

    # set mious of classes that were never in the predictions/gts to nan
    class_iou_base[~torch.isin(torch.arange(19).cuda(), glob_in_clses_base.bool())] = np.nan
    class_iou_ref[~torch.isin(torch.arange(19).cuda(), glob_in_clses_ref.bool())] = np.nan

    miou_base = class_iou_base.nanmean() * 100
    miou_ref = class_iou_ref.nanmean() * 100

    acc_base = bazeline_acc.compute()
    acc_ref = refined_acc.compute()

    # print per-class results
    for i in range(19):
        print(f"Class {dataset.CLASS2CAT[i]}-{i}: Base - {class_iou_base[i]}, refined - {class_iou_ref[i]}, "
              f"improvement - {class_iou_ref[i] - class_iou_base[i]}")

    print(f"Baseline mIoU: {miou_base}")
    print(f"Refined mIoU: {miou_ref}")
    print(f"Improvement: {miou_ref - miou_base}")

    print(50 * "=")

    np.savez(save_path, class_iou_base=class_iou_base.cpu().numpy(), class_iou_ref=class_iou_ref.cpu().numpy(),
             miou_base=miou_base.cpu().numpy(), miou_ref=miou_ref.cpu().numpy(), acc_base=acc_base.cpu().numpy(),
                acc_ref=acc_ref.cpu().numpy())


def results2latex():
    results_folder = ''


if __name__ == '__main__':
    sweep_refinement_cityscapes()
    sweep_refinement_acdc()
    ablate_refinement_corruptions()
    # results2latex()
