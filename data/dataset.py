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

    def __init__(self, list_path, mode='train', insize=64, get_with_filename=False, debug=False):
        self.list_path = list_path
        self.target_dir = '/'.join(list_path.split('/')[:-1])
        self.mode = mode
        self.insize = insize
        self.get_with_filename = get_with_filename

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
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.RandomPerspective(distortion_scale=0.2, p=0.6),
                T.RandomRotation(30),
                T.Resize((insize, insize)),
                T.ToTensor(),
                T.Normalize(mean=[0., 0., 0.], std=[255., 255., 255.])
            ])
        elif self.mode == 'test':
            self.transforms = T.Compose([
                T.Resize((insize, insize)),
                T.ToTensor(),
                T.Normalize(mean=[0., 0., 0.], std=[255., 255., 255.])
            ])

    def __getitem__(self, index):
        filepath = self.filenames[index]
        image = Image.open(filepath)
        image = self.transforms(image)
        label = filepath.split('/')[-2]
        if self.get_with_filename:
            return filepath, image.float(), self.l2i[label]
        else:
            return image.float(), self.l2i[label]

    def __len__(self):
        return len(self.filenames)

    def get_classnum(self):
        return len(self.i2l)
