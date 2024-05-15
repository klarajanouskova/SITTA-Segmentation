import glob
import sys
import os
import requests
import subprocess
from pathlib import Path



class COCO:
    COCO_IMAGES_UNLABELED_2017 = "http://images.cocodataset.org/zips/unlabeled2017.zip"
    COCO_IMAGES_TRAIN_2017 = 'http://images.cocodataset.org/zips/train2017.zip'
    COCO_IMAGES_VAL_2017 = 'http://images.cocodataset.org/zips/val2017.zip'
    COCO_IMAGES_TEST_2017 = 'http://images.cocodataset.org/zips/test2017.zip'

    COCO_INFO_UNLABELED_2017 = 'http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip'
    COCO_INFO_TEST_2017 = 'http://images.cocodataset.org/annotations/image_info_test2017.zip'

    COCO_ANNOTATIONS_TRAINVAL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


class Cityscapes:
    # 1 -> gtFine_trainvaltest.zip (241MB)
    # 2 -> gtCoarse.zip (1.3GB)
    # 3 -> leftImg8bit_trainvaltest.zip (11GB)
    # 4 -> leftImg8bit_trainextra.zip (44GB)
    # 8 -> camera_trainvaltest.zip (2MB)
    # 9 -> camera_trainextra.zip (8MB)
    # 10 -> vehicle_trainvaltest.zip (2MB)
    # 11 -> vehicle_trainextra.zip (7MB)
    # 12 -> leftImg8bit_demoVideo.zip (6.6GB)
    # 28 -> gtBbox_cityPersons_trainval.zip (2.2MB)
    subset2id = {
    'TRAINVAL_IMS_FINE_GT': 1,
    'COARSE_GT': 2,
    'TRAINVAL_IMS': 3,
    'TRAINVAL_IMS_EXTRA': 4,
    'FOG_TRAINVAL_IMS': 31,
    'RAIN_TRAINVAL_IMS': 33
    }


class ACDC:
    GT_SEM_SEG ='https://acdc.vision.ee.ethz.ch/gt_trainval.zip'
    IMGS = 'https://acdc.vision.ee.ethz.ch/rgb_anon_trainvaltest.zip'


class GTA5:
    n_parts = 10

    def get_im_link(self, idx):
        return f'https://download.visinf.tu-darmstadt.de/data/from_games/data/{idx:02d}_images.zip'

    def get_gt_link(self, idx):
        return f'https://download.visinf.tu-darmstadt.de/data/from_games/data/{idx:02d}_labels.zip'


def wgetpp(url, path=''):
    # TODO add -nc, -r as params
    Path(path).mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "-nc", "-P", path, url])
    filepath = os.path.join(path, url.split('/')[-1])
    assert os.path.exists(filepath), "Something is wrong with the filename creation logic or url"
    return filepath


def download_coco(datasets_folder):
    coco = COCO()
    coco_path = os.path.join(datasets_folder, 'coco')
    # annots
    wgetpp(coco.COCO_ANNOTATIONS_TRAINVAL, coco_path)
    #     info
    wgetpp(coco.COCO_INFO_TEST_2017, coco_path)
    # images
    wgetpp(coco.COCO_IMAGES_VAL_2017, coco_path)
    wgetpp(coco.COCO_IMAGES_TEST_2017, coco_path)
    wgetpp(coco.COCO_IMAGES_TRAIN_2017, coco_path)


def download_cityscapes(datasets_folder):
    """
    Download cityscapes dataset
    :return:
    """
    import shlex

    cityscapes_folder = os.path.join(datasets_folder, 'cityscapes')

    def load_cityscapes_pswd():
        """
        Load password for your cityscapes account cityscapes, you can create one here: https://www.cityscapes-dataset.com/login/
        Then create a file cityscapes_pswd.txt and paste the password there
        :return:i
        """
        psswd_file_path = '../cityscapes_pswd.txt'
        assert os.path.exists(psswd_file_path), "Please create a file cityscapes_pswd.txt and paste your cityscapes account password there. \n You can get one at https://www.cityscapes-dataset.com/login/"
        with open(psswd_file_path, 'r') as f:
            return f.read().strip()

    def download_subset(subset):
        Path(cityscapes_folder).mkdir(parents=True, exist_ok=True)
        pswd = load_cityscapes_pswd()
        username = 'your_username'
        args = shlex.split(
            f"wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username={username}&password={pswd}&submit=Login' https://www.cityscapes-dataset.com/login/")
        subprocess.run(args)
        args = shlex.split(
            f"wget --load-cookies=cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID={cityscapes.subset2id[subset]} -P {path}")
        subprocess.run(args)

    cityscapes = Cityscapes()
    for subset in ['TRAINVAL_IMS_FINE_GT', 'TRAINVAL_IMS', 'FOG_TRAINVAL_IMS', 'RAIN_TRAINVAL_IMS']:
        download_subset(subset)


def download_acdc(datasets_folder):
    acdc_folder = os.path.join(datasets_folder, 'acdc')
    # TODO links not working :(
    acdc = ACDC()
    wgetpp(acdc.GT_SEM_SEG, acdc_folder)
    wgetpp(acdc.IMGS, acdc_folder)


def download_gta5(datasets_folder):
    gta5_folder = os.path.join(datasets_folder, 'gta5')
    gta5 = GTA5()
    for i in range(1, gta5.n_parts + 1):
        print(f"Downloading part {i}")
        wgetpp(gta5.get_im_link(i), gta5_folder)
        wgetpp(gta5.get_gt_link(i), gta5_folder)

    # unzip all in '/datagrid/TextSpotter/klara/datasets/gta5'
    zip_files = glob.glob(os.path.join(gta5_folder, '*.zip'))
    for f in zip_files:
        # -n: never overwrite existing files, -o: overwrite existing files without prompting, -d: destination directory
        subprocess.run(["unzip", "-n", f, "-d", gta5_folder])

    # # remove all zips
    for f in zip_files:
        print(f"Removing {f}")
        os.remove(f)

#     TODO checksum?


if __name__ == '__main__':
    download_coco()
