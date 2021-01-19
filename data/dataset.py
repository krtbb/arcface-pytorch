import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import pickle


class Dataset(data.Dataset):

    def __init__(self, list_path, mode='train', insize=64, debug=False):
        self.list_path = list_path
        self.target_dir = '/'.join(list_path.split('/')[:-1])
        self.mode = mode
        self.insize = insize

        with open(list_path) as f:
            self.filenames = list(map(lambda x: x.strip(), f.readlines()))
        if debug:
            self.filenames = self.filenames[:128]

        with open(os.path.join(self.target_dir, 'i2l.pickle'), 'rb') as f:
            self.i2l = pickle.load(f)
        with open(os.path.join(self.target_dir, 'l2i.pickle'), 'rb') as f:
            self.l2i = pickle.load(f)
        
        assert self.mode in ['train', 'test']
        if self.mode == 'train':
            self.transforms = T.Compose([
                T.Resize((insize, insize)),
                T.RandomRotation(30),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif self.mode == 'test':
            self.transforms = T.Compose([
                T.Resize((insize, insize)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __getitem__(self, index):
        filepath = self.filenames[index]
        image = Image.open(filepath)
        image = self.transforms(image)
        label = filepath.split('/')[-2]
        return image.float(), self.l2i[label]

    def __len__(self):
        return len(self.filenames)

    def get_classnum(self):
        return len(self.i2l)
