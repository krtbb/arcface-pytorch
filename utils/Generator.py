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
        self.insize = self.config['insize']
        self.outsize = self.config['outsize']

        if epoch < 0:
            query = os.path.join(self.log_dir, '{}*.pth'.format(self.config['model_name']))
            pths = glob(query)
            epochs = list(map(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]), pths))
            max_epoch = max(epochs)
            print('Load models in {} epochs'.format(max_epoch))
            pth_path = os.path.join(self.log_dir, '{}_{}.pth'.format(self.config['model_name'], max_epoch))
        else:
            pth_path = os.path.join(self.log_dir, '{}_{}.pth'.format(self.config['model_name'], epoch))
        
        self.model.load_state_dict(torch.load(pth_path, map_location=device), strict=False)

    def __call__(self, x):
        return self.model(x)

class MetricGenerator(object):
    def __init__(self, log_dir, class_num, epoch=-1, device='cuda'):
        self.log_dir = log_dir

        json_path = os.path.join(self.log_dir, 'train_config.json')
        with open(json_path) as f:
            self.config = json.load(f)

        print('outsize = {}'.format(self.config['outsize']))
        print('class_num = {}'.format(class_num))

        if self.config['metric_name'] == 'add_margin':
            self.metric_fc = AddMarginProduct(self.config['outsize'], class_num, s=30, m=0.35)
            _i = 2
        elif self.config['metric_name'] == 'arc_margin':
            self.metric_fc = ArcMarginProduct(self.config['outsize'], class_num, device=device, s=30, m=0.5, easy_margin=False)
            _i = 2
        elif self.config['metric_name'] == 'sphere':
            self.metric_fc = SphereProduct(self.config['outsize'], class_num, m=4)
            _i = 1
        else:
            self.metric_fc = nn.Linear(self.config['outsize'], class_num)
            _i = 1

        self.device = torch.device(device)
        self.metric_fc.to(self.device)
        self.insize = self.config['insize']
        self.outsize = self.config['outsize']

        if epoch < 0:
            query = os.path.join(self.log_dir, '{}*.pth'.format(self.config['metric_name']))
            pths = glob(query)
            epochs = list(map(lambda x: int(x.split('/')[-1].split('.')[0].split('_')[_i]), pths))
            max_epoch = max(epochs)
            print('Load metric_fc in {} epochs'.format(max_epoch))
            pth_path = os.path.join(self.log_dir, '{}_{}.pth'.format(self.config['metric_name'], max_epoch))
        else:
            pth_path = os.path.join(self.log_dir, '{}_{}.pth'.format(self.config['metric_name'], epoch))
        
        self.metric_fc.load_state_dict(torch.load(pth_path, map_location=device), strict=False)

    def __call__(self, x, y):
        return self.metric_fc(x, y)