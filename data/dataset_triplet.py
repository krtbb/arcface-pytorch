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


class DatasetTriplet(data.Dataset):
    def __init__(self, list_path, N=50000, mode='train', insize=64, get_with_filename=False, debug=False):
        self.list_path = list_path
        self.target_dir = '/'.join(list_path.split('/')[:-1])
        self.N = N
        self.mode = mode
        self.insize = insize
        self.get_with_filename = get_with_filename

        with open(list_path) as f:
            self.filenames = list(map(lambda x: x.strip(), f.readlines()))
        if debug:
            self.filenames = self.filenames[:128]
            self.N = 100

        with open(os.path.join(self.target_dir, 'i2l.pickle'), 'rb') as f:
            self.i2l = pickle.load(f)
        with open(os.path.join(self.target_dir, 'l2i.pickle'), 'rb') as f:
            self.l2i = pickle.load(f)
        
        self.filedict = {}
        for name in self.filenames:
            id = self.__get_id(name)
            if id in self.filedict.keys():
                self.filedict[id].append(name)
            else:
                self.filedict[id] = [name]
        
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

        self.shuffle_samples()
        

    def __getitem__(self, index):
        anc_path, pos_path, neg_path = self.samples[index]
        anc_image = self.__load_filename__(anc_path)
        pos_image = self.__load_filename__(pos_path)
        neg_image = self.__load_filename__(neg_path)
        return anc_image, pos_image, neg_image

    def __load_filename__(self, filename):
        image = Image.open(filename)
        image = self.transforms(image)
        return image.float()

    def __len__(self):
        return self.N

    def __get_id(self, name):
        return name.split('/')[5]

    def shuffle_samples(self, N=None):
        self.samples = []
        N = self.N if N is None else N
        #"""
        while len(self.samples) < N:
            sample_anchor = np.random.choice(self.filenames, 1)[0]
            sample_id = self.__get_id(sample_anchor)
            while True:
                sample_positive = np.random.choice(self.filedict[sample_id], 1)[0]
                if sample_anchor != sample_positive or len(self.filedict[sample_id])==1:
                    break
            sample_negative_id = np.random.choice(list(self.filedict.keys()), 1)[0]
            sample_negative = np.random.choice(self.filedict[sample_negative_id], 1)[0]
            self.samples.append([sample_anchor, sample_positive, sample_negative])
        #"""
        #samples_anchor = np.random.choices(self.filenames, k=N)
        #samples_positive = 

