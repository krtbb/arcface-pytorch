import json
import numpy as np
import os
import sys
import torch

from tqdm import tqdm

from data.dataset import Dataset
from utils.Generator import Generator, MetricGenerator

def evaluate(
        log_dir,
        eval_list,
        epoch = -1,
        batchsize = 256,
        num_workers = 4,
        device_name = 'cuda',
        debug = False
    ):

    device = torch.device(device_name)

    # load config file
    config_path = os.path.join(log_dir, 'train_config.json')
    with open(config_path) as f:
        config = json.load(f)

    # load dataset
    eval_dataset = Dataset(eval_list, mode='test', insize=config['insize'], get_with_filename=True, debug=debug)
    dataloader = torch.utils.data.DataLoader(eval_dataset,
                                    batch_size = batchsize,
                                    shuffle = False,
                                    num_workers = num_workers)
    class_num = eval_dataset.get_classnum()

    # load model and metrics
    model_gen = Generator(log_dir, epoch=epoch, device=device_name)
    metric_gen = MetricGenerator(log_dir, class_num, epoch=epoch, device=device_name)
    assert model_gen.config['outsize'] == metric_gen.config['insize']

    # run prediction
    preds_history = []
    labels_history = []

    total_num = 0.
    corrects_num = 0.
    for i, data in enumerate(tqdm(dataloader)):
        _, images, labels = data
        images = images.to(device)
        labels = labels.to(device).long()
        features = model_gen(images)
        features = features.to(device)
        probabilities = metric_gen(features, labels)
        preds = np.argmax(probabilities.data.cpu().numpy(), axis=1)
        labels_cpu = labels.data.cpu().numpy()
        acc = (preds==labels_cpu).astype(int)

        total_num += len(images) # batch num
        corrects_num += np.sum(acc) # corrent prediction num

        preds_history.extend(list(preds))
        labels_history.extend(list(labels_cpu))

    accuracy = corrects_num / total_num

    # print result
    for p, l in zip(preds_history, labels_history):
        print(p, l)

    print('** ', log_dir, ' **')
    print('  Total data: {}'.format(total_num))
    print('  Correct prediction: {}'.format(corrects_num))
    print('  Accuracy = {}'.format(accuracy))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('lists')
    parser.add_argument('--epoch', type=int, default=-1, help='target epochs of models to predict.')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda', help='target device name, default=\'cuda\'')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    evaluate(
        args.log_dir,
        args.lists,
        epoch = args.epoch,
        batchsize = args.batchsize,
        num_workers = args.num_workers,
        device_name = args.device,
        debug = args.debug
    )