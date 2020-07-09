import os
import pandas as pd
import numpy as np

from PIL import Image
from cv2 import cv2

import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils


class ThyroidNoduleDataset(Dataset):
    def __init__(self, root='./data/', split='train', transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        
        file_list = os.listdir(root + self.split + '/image/')
        self.items = [file for file in file_list if file.endswith(".PNG")]

    def __getitem__(self, index):
        if self.split == 'train':
            src = cv2.imread(root + 'train/image/' + str(self.items[index]))
            mask = cv2.imread(root + 'train/mask/' + str(self.items[index]))
        elif self.split == 'val':
            src = cv2.imread(root + 'val/image/' + str(self.items[index]))
            mask = cv2.imread(root + 'val/mask/' + str(self.items[index]))
        elif self.split == 'test':
            src = cv2.imread(root + 'test/image/' + str(self.items[index]))
            mask = cv2.imread(root + 'test/mask/' + str(self.items[index]))
        
        sample = (src, mask)

        return sample if self.transform is None else self.transform(*sample)

    def __len__(self):
        return len(self.items)
        