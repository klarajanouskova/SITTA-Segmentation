from data.coco import coco_transform, coco_transform_invert
from data.driving_dataset import DrivingDataset
from data.voc import mask2color as voc_mask2color

class BaseDataset():
    def __init__(self, base_dataset):
        assert base_dataset in ["coco", "gta5"]
        self.base_dataset = base_dataset

    def normalize(self,x,to_tensor=True):
        if self.base_dataset == "coco":
            return coco_transform(to_tensor)(x)
        else:
            return DrivingDataset.normalize(x)

    def denormalize(self,x):
        if self.base_dataset == "coco":
            return coco_transform_invert(x)
        else:
            return DrivingDataset.denormalize(x)

    def mask2color(self, mask):
        if self.base_dataset == "coco":
            return voc_mask2color(mask)
        else:
            return DrivingDataset.mask2color(mask)
