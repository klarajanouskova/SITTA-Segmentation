import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import os
import warnings

import matplotlib.pyplot as plt

import sys

sys.path.append('..')

from tqdm import tqdm
from torchvision.transforms import ToPILImage
from PIL import Image

from data.gta5 import get_gta5
from data.coco import get_coco

from data.driving_dataset import DrivingDataset
from data.coco import COCOSegmentation
from data.coco import coco_transform_invert
from distortion import distortions_cov

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DATASET = "coco"  # gta5 or coco

num_classes = 21 if BASE_DATASET == 'coco' else 19


class ValidConfig:

    distortion_keys = [
        'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur',
        'gaussian_blur', 'brightness', 'contrast', 'jpeg', 'none'
    ]
    severities = [1, 3, 5]
    # defaults
    n_samples = 40


def gen_corrupted(cfg, corr_cfg, dataset, debug=False):
    out_data_path = os.path.join(cfg.data_dir, 'TTA', f'{BASE_DATASET}_corr')
    out_im_path = os.path.join(out_data_path, 'images')
    out_gt_path = os.path.join(out_data_path, 'labels')
    Path(out_im_path).mkdir(parents=True, exist_ok=True)
    Path(out_gt_path).mkdir(parents=True, exist_ok=True)


    save_orig = True
    for severity in corr_cfg.severities:

        # set seed to 0 if dataset is coco, 1 for gta5, for consistency with the paper
        # (different people worked on each)
        if BASE_DATASET == "coco":
            cfg.seed = 0
        else:
            cfg.seed = 1
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)


        for distortion_name in corr_cfg.distortion_keys:
            if distortion_name == 'none':
                # we are saving original images anyway
                continue
            corrupt_fun = distortions_cov[distortion_name]

            print(f'Generating corrupted images for {distortion_name} with severity {severity}')
            for c in tqdm(range(corr_cfg.n_samples)):
                img, gt, _, _ = dataset[c]

                if BASE_DATASET == "coco":
                    img = coco_transform_invert(img)
                    denorm_im = COCOSegmentation.np2pil(img.numpy())
                    gt = np.array(gt)
                else:
                    denorm_im = DrivingDataset.tensor2pil(img)

                # distort and add batch dimension
                if corrupt_fun is not None:
                    dist = corrupt_fun(denorm_im, severity=severity)
                    dist = dist.astype(np.uint8)

                    dist = Image.fromarray(dist)
                else:
                    dist = denorm_im

                if debug and c == 0:
                    plt.imshow(dist)
                    plt.title(f'{distortion_name} s{severity} c{c}')
                    plt.show()
                    plt.imshow(denorm_im)
                    plt.title('Original image')
                    plt.show()

                im_save_path = os.path.join(out_im_path, f'{distortion_name}_s{severity}_{c}.png')
                dist.save(im_save_path)

                # load dist from file and check if it is the same
                if debug:
                    dist_load = Image.open(im_save_path)
                    assert (np.array(dist) != np.array(dist_load)).sum() == 0, 'Loaded image is different from saved image'

                # save orig im file name to file
                if save_orig:
                    orig_im_save_path = os.path.join(out_im_path, f'{c}.png')
                    denorm_im.save(orig_im_save_path)
                    gt_save_path = os.path.join(out_gt_path, f'{c}.png')
                    #    save the np gt as pil
                    gt = Image.fromarray(gt.astype(np.uint8))
                    gt.save(gt_save_path)
            # only save orig once
            save_orig = False



def get_dataset(cfg):
    if cfg.name == "gta5":
        dataset = get_gta5(split='val', val_size=250)
    elif cfg.name == "coco":
        dataset = get_coco(split='val')

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


@hydra.main(config_path=f"../conf/{'coco' if BASE_DATASET == 'coco' else 'gta'}", config_name="tta_eval",
            version_base=None)
def main(cfg):
    # allow adding new keys
    OmegaConf.set_struct(cfg, False)
    complete_config(cfg, print_cfg=True)
    create_output_dirs(cfg)

    # set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    dataset = get_dataset(cfg)

    cfg.seed = 1
    corr_cfg = ValidConfig()

    gen_corrupted(cfg, corr_cfg, dataset, debug=False)

if __name__ == '__main__':
    main()
