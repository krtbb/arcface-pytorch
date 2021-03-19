import json
import numpy as np
import os
import sys
import torch

from tqdm import tqdm

from data.dataset import Dataset
from utils.Generator import Generator

def calculate(
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

    # load_model
    model_gen = Generator(log_dir, epoch=epoch, device=device_name)
    
    # execute calculation
    for i, data in enumerate(tqdm(dataloader)):
        filenames, images, labels = data
        images = images.to(device)
        features_gpu = model_gen(images)
        features_npy = features_gpu.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        for filename, feature in zip(filenames, features_npy):
            output_name = filename.replace('.jpg', '_mine.npy')
            np.save(output_name, feature)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('lists')
    args = parser.parse_args()

    calculate(
        args.log_dir,
        args.lists,
        epoch = -1,
        batchsize = 2,
        num_workers = 4,
        device_name = 'cuda',
        debug = False
        )
