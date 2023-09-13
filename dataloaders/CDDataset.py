from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json

class ImageDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 2
        self.load_memory = False
        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(ImageDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split == "val":
            file_list = os.path.join(self.root, 'list', f"{self.split}" + ".txt")
        elif self.split == "test":
            file_list = os.path.join(self.root, 'list', f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        f = open(file_list, "r") 
        self.files = f.readlines() 
        f.close() 

    def _load_data(self, index):
        image_A_path, image_B_path, label_path = self.files[index].split(' ')
        image_A_path    = os.path.join(self.root, image_A_path)
        image_B_path    = os.path.join(self.root, image_B_path)
        label_path      = os.path.join(self.root, label_path).strip('\n')
        
        image_A         = np.asarray(Image.open(image_A_path), dtype=np.uint8)
        image_B         = np.asarray(Image.open(image_B_path), dtype=np.uint8)
        image_id        = image_A_path.split('\\')[-1].split(".")[0]
        label           = np.asarray(Image.open(label_path), dtype=np.int32)

        return image_A, image_B, label, [image_A_path, image_B_path, label_path]

class CDDataset(BaseDataLoader):
    def __init__(self, kwargs):
        self.MEAN       = [0.485, 0.456, 0.406]
        self.STD        = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean']  = self.MEAN
        kwargs['std']   = self.STD
        
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')
        
        self.dataset = ImageDataset(**kwargs)

        super(CDDataset, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)
