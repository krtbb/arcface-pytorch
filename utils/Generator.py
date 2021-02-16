import json
import os
import sys
import torch

from collections import OrderedDict
from glob import glob

from models import *


def fix_key(state_dict):
    # from https://qiita.com/tand826/items/fd11f84e1b015b88642e
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict

class Generator(object):
    def __init__(self, log_dir, epoch=-1, device='cuda', load_state=True):
        self.log_dir = log_dir

        json_path = os.path.join(self.log_dir, 'train_config.json')
        with open(json_path) as f:
            self.config = json.load(f)
        
        models_list = ['resnet{}'.format(n) for n in [18, 34, 50, 101, 152]]
        models_list.append('simplev1')

        assert self.config['model_name'] in models_list
        if self.config['model_name'] == 'resnet18':
            self.model = resnet_face18(self.config['insize'], self.config['outsize'])
        elif self.config['model_name'] == 'resnet34':
            self.model = resnet34(self.config['insize'], self.config['outsize'])
        elif self.config['model_name'] == 'resnet50':
            self.model = resnet50(self.config['insize'], self.config['outsize'])
        elif self.config['model_name'] == 'resnet101':
            self.model = resnet101(self.config['insize'], self.config['outsize'])
        elif self.config['model_name'] == 'resnet152':
            self.model = resnet152(self.config['insize'], self.config['outsize'])
        elif self.config['model_name'] == 'simplev1':
            self.model = CNNv1(self.config['insize'], self.config['outsize'], activation='relu', kernel_pattern='v1')
        
        self.device = torch.device(device)
        self.model.to(self.device)
        self.insize = self.config['insize']
        self.outsize = self.config['outsize']

        if load_state:
            self.load_state(epoch, strict=True)

    def load_state(self, epoch, strict=False):
        if epoch < 0:
            query = os.path.join(self.log_dir, '{}*.pth'.format(self.config['model_name']))
            pths = glob(query)
            epochs = list(map(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]), pths))
            load_epoch = max(epochs)
        else:
            load_epoch = epoch

        self.pth_path = os.path.join(self.log_dir, '{}_{}.pth'.format(self.config['model_name'], load_epoch))
        self.model.load_state_dict(fix_key(torch.load(self.pth_path, map_location=self.device)), strict=strict)
        #self.model.load_state_dict(torch.load(self.pth_path, map_location=self.device), strict=strict)
        print('Load encoding model in {} epochs'.format(load_epoch))
    
    def __call__(self, x):
        return self.model(x)

class MetricGenerator(object):
    def __init__(self, log_dir, class_num, epoch=-1, device='cuda', load_state=True):
        self.log_dir = log_dir

        json_path = os.path.join(self.log_dir, 'train_config.json')
        with open(json_path) as f:
            self.config = json.load(f)

        if self.config['metric_name'] == 'add_margin':
            self.metric_fc = AddMarginProduct(self.config['outsize'], class_num, s=30, m=0.35)
        elif self.config['metric_name'] == 'arc_margin':
            self.metric_fc = ArcMarginProduct(self.config['outsize'], class_num, device=device, s=30, m=0.5, easy_margin=False)
        elif self.config['metric_name'] == 'sphere':
            self.metric_fc = SphereProduct(self.config['outsize'], class_num, m=4)
        else:
            self.metric_fc = nn.Linear(self.config['outsize'], class_num)

        self.device = torch.device(device)
        self.metric_fc.to(self.device)
        self.insize = self.config['insize']
        self.outsize = self.config['outsize']

        if load_state:
            self.load_state(epoch, strict=True)

    def load_state(self, epoch, strict=False):
        if epoch < 0:
            query = os.path.join(self.log_dir, '{}*.pth'.format(self.config['metric_name']))
            pths = glob(query)
            _i = self.config['metric_name'].count('_') + 1
            epochs = list(map(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[_i]), pths))
            load_epoch = max(epochs)
        else:
            load_epoch = epoch

        self.pth_path = os.path.join(self.log_dir, '{}_{}.pth'.format(self.config['metric_name'], load_epoch))
        self.metric_fc.load_state_dict(fix_key(torch.load(self.pth_path, map_location=self.device)), strict=strict)
        #self.metric_fc.load_state_dict(torch.load(self.pth_path, map_location=self.device), strict=strict)
        print('Load metric model in {} epochs'.format(load_epoch))
    
    def __call__(self, x, y):
        return self.metric_fc(x, y)