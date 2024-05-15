# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_tta_model(path, model_class, num_classes=1, max_size=-1):
    ckpt = torch.load(path, map_location=device)

    model = model_class(channels=num_classes, max_size=max_size)
    model.to(device)
    msg = model.load_state_dict(ckpt['model_seg'], strict=True)
    print(msg)
    return model


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(cfg):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        cfg.rank = int(os.environ['SLURM_PROCID'])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        cfg.distributed = False
        return

    cfg.distributed = True

    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        cfg.rank, cfg.dist_url, cfg.gpu), flush=True)
    torch.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                         world_size=cfg.world_size, rank=cfg.rank)
    torch.distributed.barrier()
    setup_for_distributed(cfg.rank == 0)


def init_distributed_mode_simple(args):
    # simplified, you can find proper distirbuted setup in orig repo
    print('Not using distributed mode')
    setup_for_distributed(is_master=True)  # hack
    args.distributed = False


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(cfg, epoch, model_without_ddp, optimizer, loss_scaler=None, best=False, last=False):
    # TODO simplify this
    output_dir = Path(cfg.ckpt_dir) / cfg.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / cfg.exp_name / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model_seg': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'cfg': cfg,
            }
            if best:
                save_on_master(to_save, output_dir / 'checkpoint-best.pth')
            elif last:
                save_on_master(to_save, output_dir / 'checkpoint-last.pth')
            else:
                save_on_master(to_save, checkpoint_path)
    else:
        to_save = {
            'model_seg': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': cfg,
        }
        if best:
            save_on_master(to_save, output_dir / 'checkpoint-best.pth')
        elif last:
            save_on_master(to_save, output_dir / 'checkpoint-last.pth')


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model_seg'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval_saliency):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x