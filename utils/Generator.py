import json
import os
import sys
import torch

from glob import glob

from models import *

class Generator(object):
    def __init__(self, log_dir, epoch=-1, device='cuda'):
        self.log_dir = log_dir

        json_path = os.path.join(self.log_dir, 'train_config.json')
        with open(json_path) as f:
            self.config = json.load(f)
        
        assert self.config['model_name'] in ['resnet{}'.format(n) for n in [34, 50, 101, 152]]
        if self.config['model_name'] == 'resnet34':
            self.model = resnet34(self.config['insize'], self.config['outsize'])
        elif self.config['model_name'] == 'resnet50':
            self.model = resnet50(self.config['insize'], self.config['outsize'])
        elif self.config['model_name'] == 'resnet101':
            self.model = resnet101(self.config['insize'], self.config['outsize'])
        elif self.config['model_name'] == 'resnet152':
            self.model = resnet152(self.config['insize'], self.config['outsize'])
        
        self.device = torch.device(device)
        self.model.to(self.device)

        if epoch < 0:
            query = os.path.join(self.log_dir, '{}*.pth'.format(self.config['model_name']))
            pths = glob(query)
            epochs = list(map(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]), pths))
            max_epoch = max(epochs)
            pth_path = os.path.join(self.log_dir, '{}_{}.pth'.format(self.config['model_name'], max_epoch))
        else:
            pth_path = os.path.join(self.log_dir, '{}_{}.pth'.format(self.config['model_name'], epoch))
        
        self.model.load_state_dict(torch.load(pth_path), strict=False)

    def __call__(self, x):
        return self.model(x)