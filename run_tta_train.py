import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import os

import sys

sys.path.append('')

from tqdm import tqdm
from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.transforms.functional import pil_to_tensor

from data.driving_dataset import DrivingDataset
from data.common_objs_dataset import CommonObjectsDataset
from data.coco import coco_transform
from data.corruption_val_dataset import GTA5CValDataset, COCOCValDataset
from tta import TestTimeAdaptor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValidSweepConfig:
    n_ims = [1]
    learn_methods = ['tent', 'ref', 'adv', 'pl', 'iou']

    method_lrs_iou = {
        'norm':
            {
                'tent': [1e-1, 5e-2, 1e-2, 8e-3, 5e-3],
                'adv': [1e-7, 5e-8, 1e-8],
                'iou': [5e-2, 1e-2, 5e-3, 1e-3],
                'ref': [1e-2, 5e-3, 1e-3, 5e-4],
                'pl': [5e-1, 1e-1, 5e-2, 1e-2],
                'augco': [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
            },
        'all':
            {
                'tent': [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-6],
                'adv': [1e-7, 5e-8, 1e-8],
                'iou': [1e-3, 5e-3, 1e-3],
                'ref': [5e-3, 1e-3, 5e-4, 1e-4, 5e-5],
                'pl': [5e-4, 1e-4, 5e-5],
                'augco': [1e-4, 5e-5, 3e-5, 1e-5],
            }
    }

    method_lrs_ce = {
        'norm': {
            'ref': [1e-2, 5e-3, 1e-3, 5e-4],
            'pl': [1e-2, 5e-2, 5e-3, 1e-3, 5e-4],
            'augco': [1e-2, 5e-3, 8e-3, 1e-3, 5e-4],
        },
        'all': {
            'ref': [5e-3, 1e-3, 5e-4, 1e-4, 5e-5],
            'pl': [5e-5, 1e-5, 5e-6],
            'augco': [5e-5, 1e-5, 5e-6],
        }
    }

    loss_method_lrs = {'ce': method_lrs_ce, 'iou': method_lrs_iou}
    n_iters = [10]
    optims = ['sgd']
    distortion_keys = [
        'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur',
        'gaussian_blur', 'brightness', 'contrast', 'jpeg', 'none'
    ]
    severities = [1, 3, 5]
    # defaults
    train_norm_only = {'tent': True, 'ref': False, 'iou': False, 'adv': True, 'pl': True, 'augco': True}
    n_samples = 40
    loss_name = 'iou'


class ValidLayersSweepConfig(ValidSweepConfig):
    train_norm_only = {'tent': False, 'ref': True}


class TentSweepConfig(ValidSweepConfig):
    learn_methods = ['tent']


class AdvSweepConfig(ValidSweepConfig):
    learn_methods = ['adv']


class RefSweepConfig(ValidSweepConfig):
    learn_methods = ['ref']


class IouSweepConfig(ValidSweepConfig):
    learn_methods = ['iou']


class PLSweepConfig(ValidSweepConfig):
    learn_methods = ['pl']


class AugcoSweepConfig(ValidSweepConfig):
    learn_methods = ['augco']


class DummyValSweepConfig(ValidSweepConfig):
    learn_methods = ['pl']
    # adv 1e-8 good
    loss_method_lrs = {
        'ce': {
            'norm': {'tent': [1e-2], 'iou': [1e-4], 'ref': [1e-6], 'adv': [1e-9], 'pl': [5e-3], 'augco': [1e-5]},
            'all': {'tent': [1e-2], 'iou': [1e-4], 'ref': [1e-6], 'adv': [1e-9], 'pl': [5e-3], 'augco': [1e-5]}
        },
    'iou': {
        'norm': {
            'tent': [1e-2], 'iou': [1e-4], 'ref': [1e-3], 'adv': [1e-9], 'pl': [5e-4], 'augco': [1e-3]},
        'all': {
            'tent': [1e-2], 'iou': [1e-4], 'ref': [1e-3], 'adv': [1e-9], 'pl': [5e-4], 'augco': [1e-3]}
        }}
    optims = ['sgd']
    distortion_keys = [
        'fog', 'gaussian_noise'
    ]
    severities = [1, 3, 5]
    # defaults
    n_samples = 4
    n_iters = [2]
    loss_name = 'iou'

def get_dataset(cfg, distortion_name=None, severity=5):
    root = os.path.join(cfg.data_dir, 'TTA', f'{cfg.base_data.name}_corr')
    base_root = os.path.join(cfg.data_dir, 'COCO' if cfg.base_data.name == 'coco' else 'gta5')
    corr_data_class = COCOCValDataset if cfg.base_data.name == 'coco' else GTA5CValDataset
    dataset = corr_data_class(
        root=root,
        base_root=base_root,
        base_name=cfg.base_data.name,
        corruption=distortion_name,
        severity=severity
    )
    return dataset


def valid_sweep_corruptions(cfg, sweep_cfg, res_subfolder):
    for n_im in sweep_cfg.n_ims:
        for learn_method in sweep_cfg.learn_methods:
            for severity in sweep_cfg.severities:
                for lr in sweep_cfg.method_lrs[learn_method]:
                    for optim in sweep_cfg.optims:
                        for n_iter in sweep_cfg.n_iters:
                            cfg.tta.loss_name = sweep_cfg.loss_name
                            cfg.tta.n_ims = n_im
                            cfg.tta.lr = lr
                            cfg.tta.n_iter = n_iter
                            cfg.tta.optim = optim
                            cfg.tta.train_norm_only = sweep_cfg.train_norm_only[learn_method]
                            tta = TestTimeAdaptor(cfg=cfg, tta_method=learn_method, base_dataset=cfg.base_data.name)
                            eval_corrupted(cfg=cfg, tta=tta, samples=sweep_cfg.n_samples,
                                           distortion_keys=sweep_cfg.distortion_keys, severity=severity,
                                           res_subfolder=res_subfolder)


def eval_corrupted(cfg, tta, samples=20, distortion_keys=['none'], severity=5, res_subfolder='val_corr'):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    subfolder = f'tta/{cfg.base_data.name}'
    if res_subfolder is not None:
        subfolder = os.path.join(subfolder, res_subfolder)
    save_folder = os.path.join(cfg.results_dir, subfolder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    seg_ious_ours, seg_ious_corrs, seg_accs, deep_tta_losses = [], [], [], []
    seg_entropies = []
    # to compute the standard miou metric accumulated over all corruptions, not per-image
    it_iou_metrics_all = [MulticlassJaccardIndex(num_classes=cfg.base_data.num_classes, ignore_index=255, average=None) for _ in
                          range(cfg.tta.n_iter + 1)]
    # standard miou across all distortions, separately
    it_iou_metrics_corr = [MulticlassJaccardIndex(num_classes=cfg.base_data.num_classes, ignore_index=255, average=None) for _ in
                           range(cfg.tta.n_iter + 1)]

    for distortion_name in distortion_keys:
        corr_dataset = get_dataset(cfg, distortion_name, severity)

        # per-class iou losses, tta losses
        dist_seg_ious, dist_seg_accs, dist_deep_tta_losses = [], [], []
        dist_seg_entropies = []
        im_tensors, gts = [], []
        # if n_samples are changed, we need to check the subdataset is good (no problematic gt)!
        for c, data in tqdm(enumerate(corr_dataset)):
            img, im_gt, _, _ = data
            # convert pil to tensor
            im_gt = torch.tensor(np.array(im_gt), dtype=torch.int32)

            if cfg.base_data.name == "coco":
                img = CommonObjectsDataset.pil2tensor(img)
            else:
                img = DrivingDataset.pil2tensor(img)

            # add batch dimension
            im_tensor = img[None].float().to(device)


            im_tensors.append(im_tensor)
            gts.append(im_gt)

            # when a batch is accumulated, run segmentation and TTA
            if (c + 1) % cfg.tta.n_ims == 0:
                im_tensors = torch.vstack(im_tensors)
                #  bs x it x c x h x w
                xs_tta, preds_seg, loss_dict, _ = tta(im_tensors)

                # evaluate segmentation
                # Make sure it is the right shape
                for im_preds_seg, im_gt in zip(preds_seg, gts):
                    im_seg_ious, im_seg_accs = [], []
                    for it_idx, im_preds_seg_it in enumerate(im_preds_seg):
                        # argmax over classes
                        im_preds_seg_it = torch.argmax(im_preds_seg_it, dim=0)[None]
                        im_it_seg_iou = it_iou_metrics_all[it_idx](im_preds_seg_it, im_gt[None])

                        im_it_seg_acc = multiclass_accuracy(preds=im_preds_seg_it,
                                                            target=im_gt[None], ignore_index=255,
                                                            num_classes=cfg.base_data.num_classes,
                                                            average='weighted')
                        it_iou_metrics_corr[it_idx](im_preds_seg_it, im_gt[None])
                        # set to nan if class not in gt and not in pred
                        in_clses = torch.hstack([torch.unique(im_preds_seg_it), torch.unique(im_gt)]).unique()
                        im_it_seg_iou[~np.isin(np.arange(cfg.base_data.num_classes), in_clses)] = np.nan
                        im_seg_ious.append(im_it_seg_iou)
                        im_seg_accs.append(im_it_seg_acc)

                    im_seg_ious = torch.stack(im_seg_ious)
                    im_seg_accs = torch.stack(im_seg_accs)

                    # its x classes
                    im_seg_ious = im_seg_ious.numpy()
                    im_seg_accs = im_seg_accs.numpy()
                    print(f'\n NA: {np.nanmean(im_seg_ious[0]) * 100}, TTA: {np.nanmean(im_seg_ious[-1]) * 100}')

                    dist_seg_ious.append(im_seg_ious)
                    dist_seg_accs.append(im_seg_accs)
                    # sum losses from loss dict and add to dist_deep_tta_losses
                tmp_losses = []
                #     TODO fix this
                for k, v in loss_dict.items():
                    #  iter x nim, add extra dimension if necessary (nim=1)
                    tmp_losses.append(v if len(v[0].shape) > 0 else [val[None] for val in v])
                # loss x iter x nim -> nim x iter
                dist_deep_tta_losses.extend(np.array(tmp_losses).sum(0).T)
                # compute the entropy of the predictions
                ents = torch.sum(torch.special.entr(preds_seg), 2).mean((2, 3))
                dist_seg_entropies.extend(ents.cpu().numpy())


                #     reset
                im_tensors, gts = [], []

            if c == samples - 1:
                break

        # nim x iter, nim x iter x classes
        deep_tta_losses.append(dist_deep_tta_losses)
        seg_ious_ours.append(dist_seg_ious)
        seg_ious_corrs.append(
            [it_iou_metrics_corr[it_idx].compute().cpu().numpy() for it_idx in range(cfg.tta.n_iter + 1)])
        seg_accs.append(dist_seg_accs)
        seg_entropies.append(dist_seg_entropies)

        # reset the per-corruption metrics
        for it_idx in range(cfg.tta.n_iter + 1):
            it_iou_metrics_corr[it_idx].reset()

    # convert all results to np arrays (kind x im x iter),  (kind x im x iter x classes)
    deep_tta_losses = np.array(deep_tta_losses).astype(float)  # make sure None becomes nan so that we can use nanmean
    # kind x iter x class
    seg_ious_ours = np.array(seg_ious_ours) * 100
    seg_accs = np.array(seg_accs) * 100
    seg_ious_standard = [it_iou_metrics_all[it_idx].compute().cpu().numpy() for it_idx in range(cfg.tta.n_iter + 1)]
    # it x class -> class x it
    seg_ious_standard = np.array(seg_ious_standard).T * 100
    # kind x it x class -> kind x class x it
    seg_ious_corrs = np.array(seg_ious_corrs) * 100

    print('-' * 100)
    print(f'Average iou before TTA: {np.nanmean(seg_ious_ours[:, :, 0])};'
          f' after TTA: {np.nanmean(seg_ious_ours[:, :, -1])}')
    # kind x im x iter
    # print(seg_iou_losses[:, :, 0])
    # print(seg_iou_losses[:, :, -1])
    print(f'seed check: {np.nanmean(seg_ious_ours[:, :, 0])}')
    print('-' * 100)

    # precompute some aggregated results, keep iteration dimension
    miou_stand = np.nanmean(seg_ious_standard, axis=0)
    # miou_ours - first per-class mean over samples and kinds, then over classes (kind x im x iter x classes)
    miou_ours = np.nanmean(np.nanmean(seg_ious_ours, axis=(0, 1)), axis=-1)
    acc_all = seg_accs.mean(axis=(0, 1))

    #   save numpy array with results
    save_name = f'{tta.save_name}_sev_{severity}'
    print(save_name)
    # array should be kind x im x iter, kind x im x iter x classes
    save_path = os.path.join(save_folder, f'{save_name}.npz')
    print('saved')
    np.savez(save_path, deep_losses=deep_tta_losses, ious_ours=seg_ious_ours, accs=seg_accs,
             ious_standd=seg_ious_standard, ious_stand_corr=seg_ious_corrs,
             miou_stand=miou_stand, miou_ours=miou_ours, acc_all=acc_all, entropies=seg_entropies)


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

@hydra.main(config_path=f"conf_new", config_name="tta_eval",
            version_base=None)
def main_val(cfg):
    # print cfg
    print('Input config:')
    print(OmegaConf.to_yaml(cfg))
    # allow adding new keys
    OmegaConf.set_struct(cfg, False)
    complete_config(cfg, print_cfg=True)
    create_output_dirs(cfg)

    # set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    method2sweep = {'tent': TentSweepConfig, 'pl': PLSweepConfig, 'augco': AugcoSweepConfig,
                    'ref': RefSweepConfig, 'iou': IouSweepConfig, 'adv': AdvSweepConfig}

    method = cfg.tta.method
    sweep_cfg = method2sweep[method]
    sweep_cfg.train_norm_only[method] = cfg.tta.train_norm_only
    sweep_cfg.loss_name = cfg.tta.loss_name

    norm_lr_key = 'norm' if sweep_cfg.train_norm_only[method] else 'all'
    sweep_cfg.method_lrs = sweep_cfg.loss_method_lrs[sweep_cfg.loss_name][norm_lr_key]
    valid_sweep_corruptions(cfg, sweep_cfg, res_subfolder='val_corr_40/')

if __name__ == '__main__':
    main_val()
