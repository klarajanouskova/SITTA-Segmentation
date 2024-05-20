# Single Image Test-Time Adaptation for Segmentation
Official repository.
![Visualization of segmentation prediction over TTA iterations](https://github.com/klarajanouskova/SITTA-Segmentation/assets/12472166/c2d94bc0-957e-44fc-bcef-375605c64fbc)

## Software
The conda environment the code has been tested with has been exported to env.yaml, 
it can be created with:

```     
conda env create -f env.yaml
```

The environment can then be activated with:

```  
conda activate sitta
```  

We are not aware of any restrictions to some specific library versions 
but the new **torchvision v2 transforms** are used.

[//]: # (**Wandb** is used for logging the training runs - you will need to login to your account to use it. )

[//]: # (Note that you do nto need wandb if you do not want to train the refinement/iou estimation models.)


## Data and pretrained models

### Pretrained segmentation models
The pretrained COCO-segmentation model is downloaded automatically from the torchvision model zoo.
The GTA5 model can be downloaded from the
['On the Road to Online Adaptation for Semantic Image Segmentation' repository](https://github.com/naver/oasis)
(we used the DR2 checkpoint).

### TTA training synthetic datasets
GTA5-C and COCO-C datasets (synthetic datasets based on GTA5 and COCO used for TTA training) 
can be downloaded from
[Google Drive](https://drive.google.com/file/d/1nmKufy-qgA3OdcaGSaUP_em1kJCp5mY4/view?usp=drive_link)

These datasets were created based on the
['Benchmarking Neural Network Robustness to Common Corruptions and Perturbations' repository](https://github.com/hendrycks/robustness).

### Pretrained models for TTA-REF and TTA-IOU methods
The pretrained mask-refinement and iou-estimation checkpoints - you only need these models if you want to
run the 'ref' or 'iou' tta methods. Downlad from
[Google Drive](https://drive.google.com/file/d/166Lbfl1Xbum35vnn7hB6Tvz8QmkIpAaf/view?usp=drive_link).

### Test datasets   
All the datasets can be downloaded using the `download.py` script - 
note that the cityscapes dataset download requires you to have an account and provide your username and password.

[//]: # (## TTA Training)

[//]: # ()
[//]: # (You will  need to change a few things in the config:)

[//]: # (* directories - where to find datasets, save/load checkpoints and results in)

[//]: # (`conf/defaults.yaml`)

[//]: # (* your wandb credentials in `conf/wandb/wandb.yaml`)


### TTA evaluation

You will  need to change a few things in the config:
* directories - where to find datasets, save/load checkpoints and results in
`conf/defaults.yaml`

[//]: # (* your wandb credentials in `conf/wandb/wandb.yaml`)
