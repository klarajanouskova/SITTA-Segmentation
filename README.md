# Test-Time Adaptation for Segmentation

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


Adapted from the official repositories of the following papers
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- [On the Road to Online Adaptation for Semantic Image Segmentation](arxiv.org/abs/2203.16195) https://github.com/naver/oasis/tree/master

## Software
The conda environment the code has been tested with has been exported to env_tta.yml, 
it can be created with:

```     
conda env create -f env_tta.yml
```

We are not aware of any restrictions to some specific library versions 
but the new **torchvision v2 transforms** are used.

**Wandb** is used for logging and for hparameter sweeps - you will need to login to your account to use it - not
necessary but recommended, it is super convenient!

**Hydra** is used to create the experiment configurations.
You will  need to change a few things:
* directories - where to find datasets, save checkpoints and results in
`conf/path/path.yaml`
* wandb credentials in `conf/wandb/wandb.yaml`

## Downloading data and pretrained models

GTA5-C and COCO-C datasets (synthetic datasets based on GTA5 and COCO used for TTA training) 
can be downloaded from the following link:
https://drive.google.com/file/d/1nmKufy-qgA3OdcaGSaUP_em1kJCp5mY4/view?usp=drive_link

GTA5 model experiments:
The GTA5 dataset and pretrained model: []
The Cityscapes and ACDC datasets: []

COCO model experiments:
The COCO dataset and pretrained model: []
VOC dataset: []

Pretrained models

TODO, a lot is in the download script, but some dataset need to be downloaded manually.

## Structure

There are three submodules in this repository:
- **point_seg** for point-guided instance segmentation TTA with the Segment Anything model
- **sem_seg** for semantic segmentation TTA with any pretrained models
- **sem_seg_rec** (TODO) for training the joint reconstruction and segmentation model



## Training

The training can be run by launching the scripts starting with the `main_` or `main_` prefix,
which contain model and dataset loading. The code for each train and validation epoch is then
contained in the files starting wtih the `engine_` prefix.

The following scripts have beeen tested recently and should work, others may need modification to
work with the latest version of the code:

* `main_finetune_seg.py` - joint finetuning of both segmentation and reconstruction, currently on the pascal VOC dataset
* `main_train_seg_loss.py` - training of the deep  (at test time) self-supervised segmentation loss.

The code is meant to be run in distributed mode, an example of segmentation training on 2 GPUs
on a single node:

```
python -m torch.distributed.launch --nproc_per_node=2 main_finetune_seg.py --world_size 2
```

### Sweeps

It is recommended to use WandB sweeps to run hyperparameter sweeps.

You can initialize it from a config file, e.g.:

```
wandb sweep config/seg_ft_config.yml
```

and then run the sweep with:

```
wandb agent <sweep_id>
```

Note that you can launch multiple agents on different nodes once 
the sweep was initialized.

For more information about sweeps, please check the [WandB documentation](https://docs.wandb.com/sweeps).

### Evaluation
TODO