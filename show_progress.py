import os
import pandas as pd
import sys

def show_progress(target_dir, show=False, ylim=[0,0], ylog=False):
    if '*' in target_dir:
        from glob import glob
        target_dirs = glob(target_dir)
        for target_dir in sorted(target_dirs):
            show_progress(target_dir, show=show, ylim=ylim, ylog=ylog)

    else:
        if not show:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print(target_dir)
        history_path = os.path.join(target_dir, 'history.csv')
        hist = pd.read_csv(history_path)

        plt.figure(figsize=(8,6))

        plt.subplot(2,1,1)
        plt.plot(hist['epoch'], hist['train_loss'], label='train')
        plt.plot(hist['epoch'], hist['test_loss'], label='test')
        plt.grid()
        plt.ylabel('Loss')
        if int(ylim[0])**2+int(ylim[1])**2!=0: plt.ylim(ylim[0], ylim[1])
        if ylog: plt.yscale('log')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(hist['epoch'], hist['train_acc'], label='train')
        plt.plot(hist['epoch'], hist['test_acc'], label='test')
        plt.grid()
        plt.ylabel('Accuracy')
        plt.ylim(-0.1,1.1)
        plt.legend()

        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(target_dir, 'history.png'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str)

    args = parser.parse_args()

    show_progress(args.target)