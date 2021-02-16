from __future__ import print_function
import os
from data.dataset import Dataset
import torch
import torch.nn.functional as F
from models import *
import torchvision
from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *
import datetime
from tqdm import tqdm, trange
import json
from hyperdash import Experiment

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def train(
        train_list,
        test_list,
        lr,
        epoch,
        batchsize,
        insize,
        outsize,
        save_interval = 10,
        weight_decay = 5e-4,
        lr_step = 10,
        model_name = 'resnet34',
        loss_name = 'focal_loss',
        metric_name = 'arc_margin',
        optim_name = 'adam',
        num_workers = 4,
        print_freq = 1e+6,
        debug = False
    ):

    device = torch.device("cuda")

    train_dataset = Dataset(train_list, mode='train', insize=insize, debug=debug)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=num_workers)
    test_dataset = Dataset(test_list, mode='test', insize=insize, debug=debug)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                  batch_size=batchsize,
                                  shuffle=False,
                                  num_workers=num_workers)
    class_num = train_dataset.get_classnum()

    print('{} train iters per epoch:'.format(len(trainloader)))
    print('{} test iters per epoch:'.format(len(testloader)))

    if loss_name == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if model_name == 'resnet18':
        model = resnet_face18(insize, outsize)
    elif model_name == 'resnet34':
        model = resnet34(insize, outsize)
    elif model_name == 'resnet50':
        model = resnet50(insize, outsize)
    elif model_name == 'resnet101':
        model = resnet101(insize, outsize)
    elif model_name == 'resnet152':
        model = resnet152(insize, outsize)
    elif model_name == 'shuffle':
        model = ShuffleFaceNet(outsize)
    elif model_name == 'simplev1':
        model = CNNv1(insize, outsize, activation='relu', kernel_pattern='v1')
    else:
        raise ValueError('Invalid model name: {}'.format(model_name))

    if metric_name == 'add_margin':
        metric_fc = AddMarginProduct(outsize, class_num, s=30, m=0.35)
    elif metric_name == 'arc_margin':
        metric_fc = ArcMarginProduct(outsize, class_num, s=30, m=0.5, easy_margin=False)
    elif metric_name == 'sphere':
        metric_fc = SphereProduct(outsize, class_num, m=4)
    else:
        metric_fc = nn.Linear(outsize, class_num)

    # view_model(model, opt.input_shape)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    assert optim_name in ['sgd', 'adam']
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=lr, weight_decay=weight_decay)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=0.1)

    start = time.time()
    training_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyperdash_exp = Experiment(training_id)
    checkpoints_dir = os.path.join('logs', training_id)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    logging_path = os.path.join(checkpoints_dir, 'history.csv')
    
    config = {}
    config['train_list'] = train_list
    config['test_list'] = test_list
    config['lr'] = lr
    config['epoch'] = epoch
    config['batchsize'] = batchsize
    config['insize'] = insize
    config['outsize'] = outsize
    config['save_interval'] = save_interval
    config['weight_decay'] = weight_decay
    config['lr_step'] = lr_step
    config['model_name'] = model_name
    config['loss_name'] = loss_name
    config['metric_name'] = metric_name
    config['optim_name'] = optim_name
    config['num_workers'] = num_workers
    config['debug'] = debug
    for k, v in config.items():
        hyperdash_exp.param(k, v, log=False)
    with open(os.path.join(checkpoints_dir, 'train_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    with open(logging_path, 'w') as f:
        f.write('epoch,time_elapsed,train_loss,train_acc,test_loss,test_acc\n')
        
    prev_time = datetime.datetime.now()
    for i in range(epoch):
        model.train()
        for ii, data in enumerate(tqdm(trainloader, disable=True)):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            pred_classes = np.argmax(output.data.cpu().numpy(), axis=1)
            acc = np.mean((pred_classes == label.data.cpu().numpy()).astype(int))
            optimizer.zero_grad()
            loss.backward()

            #import pdb; pdb.set_trace()
            optimizer.step()
            #scheduler.step()

            iters = i * len(trainloader) + ii

            if iters % print_freq == 0 or debug:
                speed = print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))

                start = time.time()

        model.eval()
        for ii, data in enumerate(tqdm(testloader, disable=True)):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            test_loss = criterion(output, label)
            output = np.argmax(output.data.cpu().numpy(), axis=1)
            test_acc = np.mean((output == label.data.cpu().numpy()).astype(int))
            #test_acc = np.mean((torch.argmax(output, dim=1) == label).type(torch.int32))

        if i % save_interval == 0 or i == epoch:
            save_model(model.module, checkpoints_dir, model_name, i)
            save_model(metric_fc.module, checkpoints_dir, metric_name, i)

        new_time = datetime.datetime.now()
        with open(logging_path, 'a') as f:
            f.write('{},{},{},{},{},{}\n'.format(i, (new_time-prev_time).total_seconds(), loss.item(), acc, test_loss.item(), test_acc))
        prev_time = datetime.datetime.now()

        hyperdash_exp.metric('train_loss', loss.item(), log=False)
        hyperdash_exp.metric('train_acc', acc, log=False)
        hyperdash_exp.metric('test_loss', test_loss.item(), log=False)
        hyperdash_exp.metric('test_acc', test_acc, log=False)

    hyperdash_exp.end()
    print('Finished {}'.format(training_id))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('train_list')
    parser.add_argument('test_list')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epoch', default=10000, type=int)
    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--insize', default=64, type=int, help='size of input image, default=64')
    parser.add_argument('--outsize', default=128, type=int, help='size of encodings, default=128')
    parser.add_argument('--save_interval', default=10, type=int, help='about model, default=10')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay, default=5e-4')
    parser.add_argument('--lr_step', default=10, type=int)
    parser.add_argument('--model_name', default='resnet34', type=str, help='resnet34, resnet50, resnet101, resnet152')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers, default=4')
    parser.add_argument('--print_freq', default=1e+3, type=int, help='frequency of printing results, default=1e+3')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    train(
        train_list = args.train_list,
        test_list = args.test_list,
        lr = args.lr,
        epoch = args.epoch,
        batchsize = args.batchsize,
        insize = args.insize,
        outsize = args.outsize,
        save_interval = args.save_interval,
        weight_decay = args.weight_decay,
        lr_step = args.lr_step,
        model_name = args.model_name,
        num_workers = args.num_workers,
        print_freq = args.print_freq,
        debug = args.debug
    )