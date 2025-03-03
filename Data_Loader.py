from __future__ import print_function, division
import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
import random
import numpy as np
from utils import tomasks, readColormap


class Images_Dataset_folder(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        tx = Transformed images
        lx = Transformed labels"""

    def __init__(self, images_dir, labels_dir,transformI = None, transformM = None):
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM
        self.colormap = readColormap("./colormap.txt")

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
              #  torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop((500, 1000)),
               # torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.0], std=[0.5])
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
              #  torchvision.transforms.Resize((128,128)),
                torchvision.transforms.CenterCrop((500, 1000)),
                torchvision.transforms.ToTensor(),
                #torchvision.transforms.Lambda(lambda x: torch.cat([x, 1 - x], dim=0))
            ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, i):
        i1_path = os.path.join(self.images_dir, self.images[i])
        l1_path = os.path.join(self.labels_dir, self.labels[i])
        i1 = Image.open(i1_path)
        l1 = Image.open(l1_path)
        seed = np.random.randint(0, 2**31 - 1)

        # apply this seed to img tranfsorms
        random.seed(seed) 
        torch.manual_seed(seed)
        img = self.tx(i1)
        
        # apply this seed to target/label tranfsorms  
        random.seed(seed) 
        torch.manual_seed(seed)
        label = tomasks(self.lx(l1), self.colormap)

        
        return img, label