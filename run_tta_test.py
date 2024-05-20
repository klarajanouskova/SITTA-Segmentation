import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import os
import warnings

import matplotlib.pyplot as plt

import sys

sys.path.append('')

from tqdm import tqdm
from torchmetrics.functional.classification import multiclass_jaccard_index, multiclass_accuracy
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassF1Score

from data.cityscapes import get_cityscapes
from data.gta5 import get_gta5
from data.acdc import get_acdc, ACDC
from data.coco import get_coco
from data.voc import get_voc
from tta import TestTimeAdaptor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestConfigGTA5():
    # sweep with optimal parameters found in validation
    learn_methods = ['ref', 'tent', 'augco', 'pl']
    method_lr = {'tent': 1e-2, 'ref': 5e-5, 'pl': 5e-5, 'augco': 3e-5}
    method_loss_name = {'tent': 'iou', 'ref': 'iou', 'pl': 'iou', 'augco': 'iou'}
    method_n_iter = {'tent': 6, 'ref': 9, 'pl': 8, 'augco': 7}
    method_norm_only = {'tent': True, 'ref': False, 'adv': False, 'iou': True, 'pl': False, 'augco': False}
    optim = 'sgd'
    # defaults
    n_samples = 500

class TestConfigCOCO2():
    # sweep with optimal parameters found in validation
    learn_methods = ['ref', 'tent', 'augco', 'pl']
    method_lr = {'tent': 1e-2, 'ref': 5e-4, 'adv': 1e-8, 'iou': 1e-4, 'pl': 1e-4, 'augco': 5e-5}
    method_loss_name = {'tent': 'iou', 'ref': 'iou', 'adv': 'ref', 'iou': 'iou', 'pl': 'iou', 'augco': 'iou'}
    method_n_iter = {'tent': 6, 'ref': 2, 'adv': 10000, 'iou': 10000, 'pl': 5, 'augco': 5}
    method_norm_only = {'tent': True, 'ref': False, 'adv': False, 'iou': False, 'pl': False, 'augco': False}
    optim = 'sgd'
    # defaults
    train_norm_only = {'tent': True, 'ref': False, 'iou': False}
    n_samples = 500


class TestConfigCOCO():
    # sweep with optimal parameters found in validation
    learn_methods = ['ref', 'tent', 'augco', 'pl']
    method_lr = {'tent': 5e-4, 'ref': 5e-4, 'adv': 1e-8, 'iou': 1e-4, 'pl': 5e-2, 'augco': 5e-3}
    method_loss_name = {'tent': 'iou', 'ref': 'iou', 'adv': 'ref', 'iou': 'iou', 'pl': 'iou', 'augco': 'iou'}
    method_n_iter = {'tent': 2, 'ref': 5, 'adv': 10000, 'iou': 10000, 'pl': 8, 'augco': 10}
    method_norm_only = {'tent': False, 'ref': False, 'adv': False, 'iou': False, 'pl': True, 'augco': True}
    optim = 'sgd'
    # defaults
    train_norm_only = {'tent': True, 'ref': False, 'iou': False}
    n_samples = 500


def test_sweep(cfg, sweep_cfg, dataset, res_subfolder):
    for learn_method in sweep_cfg.learn_methods:
        lr = sweep_cfg.method_lr[learn_method]
        n_iter = sweep_cfg.method_n_iter[learn_method]
        cfg.tta.loss_name = sweep_cfg.method_loss_name[learn_method]
        cfg.tta.lr = lr
        cfg.tta.n_iter = n_iter
        cfg.tta.optim = sweep_cfg.optim
        cfg.tta.train_norm_only = sweep_cfg.method_norm_only[learn_method]
        tta = TestTimeAdaptor(cfg=cfg, tta_method=learn_method, base_dataset=cfg.base_data.name)
        test_tta(cfg=cfg, tta=tta, dataset=dataset, samples=sweep_cfg.n_samples, res_subfolder=res_subfolder)


def test_tta(cfg, tta, dataset, samples, res_subfolder=None):
    """
    Analysis of pseudo-mask vs prediction miou evolution over TTA iterations, both for single image and aggregated
    :param cfg:
    :param tta:
    :param dataset:
    :param samples:
    :param res_subfolder:
    :return:
    """
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    subfolder = f'tta/{cfg.test_dataset}'
    if res_subfolder is not None:
        subfolder = os.path.join(subfolder, res_subfolder)
    folder = os.path.join(cfg.results_dir, subfolder)
    Path(folder).mkdir(parents=True, exist_ok=True)

    # per-class iou losses, tta losses
    seg_ious_ours, seg_accs, deep_tta_losses = [], [], []
    seg_entropies = []
    im_tensors, gts = [], []

    # to compute the standard miou metric accumulated over all corruptions, not per-image
    ignore_index = 255
    it_iou_metrics_all = [MulticlassJaccardIndex(num_classes=cfg.base_data.num_classes, ignore_index=ignore_index, average=None) for _ in
                          range(cfg.tta.n_iter + 1)]
    it_total_acc_weighted = [MulticlassAccuracy(num_classes=cfg.base_data.num_classes, ignore_index=ignore_index, average='micro') for _ in
                            range(cfg.tta.n_iter + 1)]
    it_dice = [MulticlassF1Score(num_classes=cfg.base_data.num_classes, ignore_index=ignore_index, average='macro') for _ in
                            range(cfg.tta.n_iter + 1)]


    # if n_samples are changed, we need to check the subdataset is good (no problematic gt)!
    for c in tqdm(range(samples)):
        if cfg['base_data']['name'] == "voc":
            img, im_gt = dataset[c]
            img = img.permute(2, 0, 1)
            im_gt = np.array(im_gt)
        else:
            img, im_gt, _, _ = dataset[c]

        im_gt = torch.tensor(im_gt).int()
        im_tensors.append(img[None].to(device))
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
                # TODO move it dimension to batch dimension?
                for it_idx, im_preds_seg_it in enumerate(im_preds_seg):
                    # argmax over classes
                    im_preds_seg_it = torch.argmax(im_preds_seg_it, dim=0)[None]
                    im_it_seg_iou = it_iou_metrics_all[it_idx](im_preds_seg_it, im_gt[None])

                    im_it_seg_acc = multiclass_accuracy(preds=im_preds_seg_it,
                                                        target=im_gt[None], ignore_index=ignore_index,
                                                        num_classes=cfg.base_data.num_classes,
                                                        average='weighted')
                    im_it_total_acc_weighted = it_total_acc_weighted[it_idx](im_preds_seg_it, im_gt[None])
                    im_it_dice = it_dice[it_idx](im_preds_seg_it, im_gt[None])
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
                print(
                    f'\n NA: {np.nanmean(im_seg_ious[0]) * 100}, TTA: {np.nanmean(im_seg_ious[-1]) * 100}')

                seg_ious_ours.append(im_seg_ious)
                seg_accs.append(im_seg_accs)
                # sum losses from loss dict and add to dist_deep_tta_losses
            tmp_losses = []
            #     TODO fix this
            for k, v in loss_dict.items():
                #  iter x nim, add extra dimension if necessary (nim=1)
                tmp_losses.append(v if len(v[0].shape) > 0 else [val[None] for val in v])
            # loss x iter x nim -> nim x iter
            deep_tta_losses.extend(np.array(tmp_losses).sum(0).T)
            ents = torch.sum(torch.special.entr(preds_seg), 2).mean((2, 3))
            seg_entropies.append(ents.cpu().numpy())

            #     reset
            im_tensors, gts = [], []

    # convert all results to np arrays (kind x im x iter),  (kind x im x iter x classes)
    deep_tta_losses = np.array(deep_tta_losses).astype(float)  # make sure None becomes nan so that we can use nanmean
    # kind x iter x class
    seg_ious_ours = np.array(seg_ious_ours) * 100
    seg_accs = np.array(seg_accs) * 100
    # iou has reduction none so that we can use per-im per-class results
    miou_stand = [np.nanmean(it_iou_metrics_all[it_idx].compute().cpu().numpy()) * 100 for it_idx in range(cfg.tta.n_iter + 1)]
    total_acc_stand = [it_total_acc_weighted[it_idx].compute().cpu().numpy() * 100 for it_idx in range(cfg.tta.n_iter + 1)]
    dice_stand = [it_dice[it_idx].compute().cpu().numpy() * 100 for it_idx in range(cfg.tta.n_iter + 1)]

    # kind x it x class -> kind x class x it
    seg_entropies = np.array(seg_entropies)

    print('-' * 100)
    # print(f'Average iou before TTA: {np.nanmean(seg_ious_ours[:, :, 0])};'
    #       f' after TTA: {np.nanmean(seg_ious_ours[:, :, -1])}')
    # kind x im x iter
    print('-' * 100)

    # miou_ours - first per-class mean over samples and kinds, then over classes (kind x im x iter x classes)
    miou_ours = np.nanmean(np.nanmean(seg_ious_ours, axis=(0, 1)), axis=-1)
    acc_all = seg_accs.mean(axis=(0, 1))

    #   save numpy array with results
    save_name = f'{tta.save_name}'
    print(save_name)
    # array should be im x iter, im x iter x classes
    save_path = os.path.join(folder, f'{save_name}.npz')
    print(save_path)
    np.savez(save_path, deep_losses=deep_tta_losses, ious_ours=seg_ious_ours, accs=seg_accs,
             miou_stand=miou_stand, miou_ours=miou_ours, acc_all=acc_all, entropies=seg_entropies,
             total_acc_stand=total_acc_stand, dice_stand=dice_stand)


def test_tta_analysis(cfg, tta, dataset, samples, res_subfolder=None):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    subfolder = f'tta/{cfg.base_data.name}'
    if res_subfolder is not None:
        subfolder = os.path.join(subfolder, res_subfolder)
    folder = os.path.join(cfg.results_dir, subfolder)
    Path(folder).mkdir(parents=True, exist_ok=True)

    # per-class iou losses, tta losses
    seg_ious, seg_accs, deep_tta_losses, pl_ious, pl_accs = [], [], [], [], []
    im_tensors, gts = [], []

    # if n_samples are changed, we need to check the subdataset is good (no problematic gt)!
    for c in tqdm(range(samples)):
        if cfg['base_data']['name'] == "voc":
            img, im_gt = dataset[c]
            img = img.permute(2, 0, 1)
            im_gt = np.array(im_gt)
        else:
            img, im_gt, _, _ = dataset[c]

        im_gt = torch.tensor(im_gt).int()
        im_tensor = img[None].to(device)

        im_tensors.append(im_tensor)
        gts.append(im_gt)

        # when a batch is accumulated, run segmentation and TTA
        if (c + 1) % cfg.tta.n_ims == 0:
            im_tensors = torch.vstack(im_tensors)
            #  bs x it x c x h x w
            xs_tta, preds_seg, loss_dict, pseudomasks = tta(im_tensors, ret_pseudo_masks=True)

            # evaluate segmentation
            # Make sure it is the right shape
            for im_preds_seg, im_gt, im_pms in zip(preds_seg, gts, pseudomasks):
                im_seg_ious, im_seg_accs, im_pl_ious, im_pl_accs = [], [], [], []
                # TODO move it dimension to batch dimension?
                for it, (im_preds_seg_it, im_pm_it) in enumerate(zip(im_preds_seg, im_pms)):
                    # argmax over classes
                    im_preds_seg_it = torch.argmax(im_preds_seg_it, dim=0)[None]
                    im_it_seg_iou_losses = multiclass_jaccard_index(preds=im_preds_seg_it,
                                                                    target=im_gt[None],
                                                                    num_classes=cfg.base_data.num_classes,
                                                                    ignore_index=255,
                                                                    average='none')
                    im_it_pl_iou_losses = multiclass_jaccard_index(preds=im_pm_it,
                                                                   target=im_gt[None],
                                                                   num_classes=cfg.base_data.num_classes,
                                                                   ignore_index=255,
                                                                   average='none')
                    # set to nan if class not in gt and not in pred
                    in_clses_seg = torch.hstack([torch.unique(im_preds_seg_it), torch.unique(im_gt)]).unique()
                    im_it_seg_iou_losses[~np.isin(np.arange(19), in_clses_seg)] = np.nan
                    im_seg_ious.append(im_it_seg_iou_losses)

                    in_clses_pl = torch.hstack([torch.unique(im_pm_it), torch.unique(im_gt)]).unique()
                    im_it_pl_iou_losses[~np.isin(np.arange(19), in_clses_pl)] = np.nan
                    im_pl_ious.append(im_it_pl_iou_losses)

                    im_it_seg_acc = multiclass_accuracy(preds=im_preds_seg_it,
                                                        target=im_gt[None], ignore_index=255,
                                                        num_classes=cfg.base_data.num_classes,
                                                        average='weighted')
                    im_it_pl_acc = multiclass_accuracy(preds=im_pm_it,
                                                       target=im_gt[None], ignore_index=255,
                                                       num_classes=cfg.base_data.num_classes,
                                                       average='weighted')

                    im_seg_accs.append(im_it_seg_acc)
                    im_pl_accs.append(im_it_pl_acc)
                im_seg_ious = torch.stack(im_seg_ious)
                im_seg_accs = torch.stack(im_seg_accs)
                im_pl_ious = torch.stack(im_pl_ious)
                im_pl_accs = torch.stack(im_pl_accs)

                # its x classes
                im_seg_ious = im_seg_ious.numpy()
                print(
                    f'\n NA: {np.nanmean(im_seg_ious[0]) * 100}, TTA: {np.nanmean(im_seg_ious[-1]) * 100}')

                seg_ious.append(im_seg_ious)
                seg_accs.append(im_seg_accs)
                pl_ious.append(im_pl_ious)
                pl_accs.append(im_pl_accs)

                # plot the pseudolabel vs segmentation prediction mask evolution over tta iterations, all in one plot
                plt.figure(figsize=(20, 10))
                im_seg_mious = np.nanmean(im_seg_ious, axis=1) * 100
                im_pl_mious = np.nanmean(im_pl_ious, axis=1) * 100
                plt.plot(im_seg_mious, label='seg')
                plt.plot(im_pl_mious, label='pl')
                plt.legend()
                plt.xlabel('tta iteration')
                plt.ylabel('mIoU (%)')
                plt.show()

                # plot the ious for all classes separately, show each class in one color
                plt.figure(figsize=(20, 10))
                for cls in range(19):
                    plt.plot(im_seg_ious[:, cls], linestyle='--', color=f'C{cls}')
                    plt.plot(im_pl_ious[:, cls], label=dataset.CLASS2CAT[cls], linestyle='-', color=f'C{cls}')
                # add label for pl/seg linestyle
                plt.plot([], [], label='seg', linestyle='--', color='black')
                plt.plot([], [], label='pl', linestyle='-', color='black')
                plt.legend()
                plt.xlabel('tta iteration')
                plt.ylabel('IoU (%)')
                plt.show()
                print()

            tmp_losses = []
            #     TODO fix this
            for k, v in loss_dict.items():
                #  iter x nim, add extra dimension if necessary (nim=1)
                tmp_losses.append(v if len(v[0].shape) > 0 else [val[None] for val in v])
            # loss x iter x nim -> nim x iter
            deep_tta_losses.extend(np.array(tmp_losses).sum(0).T)

            #     reset
            im_tensors, gts = [], []

    # convert all results to np arrays (im x iter),  (im x iter x classes)
    deep_tta_losses = np.array(deep_tta_losses).astype(float)  # make sure None becomes nan so that we can use nanmean
    seg_ious = np.array(seg_ious) * 100
    seg_accs = np.array(seg_accs) * 100
    pl_ious = np.array(pl_ious) * 100
    pl_accs = np.array(pl_accs) * 100

    # miou_ours - first per-class mean over samples and kinds, then over classes (kind x im x iter x classes)
    seg_iou_ours = np.nanmean(seg_ious, axis=1)
    seg_miou_ours = np.nanmean(seg_iou_ours, axis=-1)

    pl_iou_ours = np.nanmean(pl_ious, axis=1)
    pl_miou_ours = np.nanmean(pl_iou_ours, axis=-1)

    #   plot the evolution of the mIoU over tta iterations for both seg predictions and pl predictions
    plt.figure(figsize=(20, 10))
    plt.plot(seg_miou_ours, label='seg')
    plt.plot(pl_miou_ours, label='pl')
    plt.legend()
    plt.xlabel('tta iteration')
    plt.ylabel('mIoU (%)')
    plt.show()

    #   plot the evolution of the IoU over tta iterations for each class separately for both seg predictions and pl predictions
    plt.figure(figsize=(20, 10))
    for cls in range(19):
        plt.plot(seg_iou_ours[:, cls], linestyle='--', color=f'C{cls}')
        plt.plot(pl_iou_ours[:, cls], label=dataset.CLASS2CAT[cls], linestyle='-', color=f'C{cls}')
    # add label for pl/seg linestyle
    plt.plot([], [], label='seg', linestyle='--', color='black')
    plt.plot([], [], label='pl', linestyle='-', color='black')
    plt.legend()
    plt.xlabel('tta iteration')
    plt.ylabel('IoU (%)')
    plt.show()
    print()

    # plot initial seg ious on the x axis and the final seg ious on the y axis
    plt.figure(figsize=(20, 10))
    plt.scatter(seg_iou_ours[0], seg_iou_ours[-1])
    plt.xlabel('initial mIoU (%)')
    plt.ylabel('final mIoU (%)')
    plt.show()
    print()


def get_dataset(cfg):
    if cfg.test_dataset == "gta5":
        dataset = get_gta5(split='val', val_size=250, data_path=cfg.data_dir)
    elif cfg.test_dataset == "cityscapes":
        dataset = get_cityscapes(split='val', condition='clean', data_path=cfg.data_dir)
    elif cfg.test_dataset == "acdc-night":
        dataset = get_acdc(condition='night', data_path=cfg.data_dir)
    elif cfg.test_dataset == "acdc-fog":
        dataset = get_acdc(condition='fog', data_path=cfg.data_dir)
    elif cfg.test_dataset == "acdc-rain":
        dataset = get_acdc(condition='rain', data_path=cfg.data_dir)
    elif cfg.test_dataset == "acdc-snow":
        dataset = get_acdc(condition='snow', data_path=cfg.data_dir)
    elif cfg.test_dataset == "coco":
        dataset = get_coco(split='val', data_path=cfg.data_dir)
    elif cfg.test_dataset == "voc":
        dataset = get_voc(split="val", data_path=cfg.data_dir)

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
    vis_dir = os.path.join(cfg.out_dir, 'vis')
    cfg.vis_dir = vis_dir

    if print_cfg:
        print(OmegaConf.to_yaml(cfg))


@hydra.main(config_path=f"conf_new", config_name="tta_eval",
            version_base=None)
def main_test(cfg):
    """
    Benchmark test datasets
    :param cfg:
    :return:
    """

    # allow adding new keys
    OmegaConf.set_struct(cfg, False)
    complete_config(cfg, print_cfg=True)
    create_output_dirs(cfg)

    # set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # make training seed deterministic
    # torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    # method = 'tent'
    method = cfg.tta.method


    # used to index different test datasets
    cfg.test_data_idxs = str(cfg.test_data_idxs)
    data_idxs = [int(s) for s in cfg.test_data_idxs.split(",")]

    test_cfg = TestConfigGTA5 if cfg.base_data.name == "gta5" else TestConfigCOCO
    test_cfg.learn_methods = [method]
    # test_cfg = DummyTestConfig

    if 0 in data_idxs:
        cfg.test_dataset = "voc" if cfg.base_data.name == "coco" else "cityscapes"
        print(f"Testing {cfg.base_data.name}")
        dataset = get_dataset(cfg)
        test_sweep(cfg, test_cfg, dataset, res_subfolder='test/')

    if not cfg.base_data.name == "coco":
        for c_idx, cond in enumerate(ACDC.VALID_CONDS):
            if c_idx + 1 not in data_idxs:
                continue
            print(f"Testing {cond}")
            cfg.test_dataset = f"acdc-{cond}"
            dataset = get_dataset(cfg)
            test_sweep(cfg, test_cfg, dataset, res_subfolder=f'test/')


if __name__ == '__main__':
    main_test()
