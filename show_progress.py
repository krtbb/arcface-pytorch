import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

def show_progress(target_dir):
    if '*' in target_dir:
        from glob import glob
        target_dirs = glob(target_dir)
        for target_dir in target_dirs:
            show_progress(target_dir)

    else:
        print(target_dir)
        history_path = os.path.join(target_dir, 'history.csv')
        hist = pd.read_csv(history_path)

        plt.figure(figsize=(8,8))

        plt.subplot(2,1,1)
        plt.plot(hist['epoch'], hist['train_loss'], label='train')
        plt.plot(hist['epoch'], hist['test_loss'], label='test')
        plt.grid()
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(hist['epoch'], hist['train_acc'], label='train')
        plt.plot(hist['epoch'], hist['test_acc'], label='test')
        plt.grid()
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, 'history.png'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str)

    args = parser.parse_args()

    show_progress(args.target)