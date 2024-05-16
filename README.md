# Single Image Test-Time Adaptation for Segmentation
Coming this week - stay tuned :)

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

The pretrained mask-refinement and iou-estimation checkpoints for the 'ref' and 'iou' tta methods: https://drive.google.com/file/d/166Lbfl1Xbum35vnn7hB6Tvz8QmkIpAaf/view?usp=drive_link

GTA5 model experiments:
The GTA5 dataset and pretrained model: []
The Cityscapes and ACDC datasets: []

COCO model experiments:
The COCO dataset and pretrained model: []
VOC dataset: []

Pretrained models

TODO, a lot is in the download script, but some dataset need to be downloaded manually.

## TTA Training

TODO

### Evaluation
TODO
