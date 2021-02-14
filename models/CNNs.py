import torch
import torch.nn as nn

class CNNv1(nn.Module):
    def __init__(self, insize, outsize, activation='relu', kernel_pattern='v1'):
        super(CNNv1, self).__init__()
        self.insize = insize
        self.outsize = outsize

        # kernel patterns
        if kernel_pattern == 'v1':
            conv_num = 3
            kss = [7, 5, 3]
            sts = [2, 2, 2]
            pds = [3, 2, 1]
        elif kernel_pattern == 'v2':
            conv_num = 3
            kss = [3, 3, 3]
            sts = [2, 2, 2]
            pds = [1, 1, 1]
        else:
            raise ValueError('Invalid kernel_pattern: {}'.format(kernel_pattern))

        # activation function
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError('Invalid kernel_pattern: {}'.format(kernel_pattern))

        # layer definition
        self.conv1 = nn.Conv2d(  3, 128, kernel_size=kss[0], stride=sts[0], padding=pds[0], bias=True) # 32
        self.conv2 = nn.Conv2d(128, 128, kernel_size=kss[1], stride=sts[1], padding=pds[1], bias=True) # 16
        self.conv3 = nn.Conv2d(128,  64, kernel_size=kss[2], stride=sts[2], padding=pds[2], bias=True) # 8
        self.flatten = nn.Flatten()
        self.fn4 = nn.Linear(8*8*64, 1024)
        self.fn5 = nn.Linear(1024, 512)
        self.fn6 = nn.Linear(512, self.outsize)

    def __call__(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.flatten(x)
        x = self.act(self.fn4(x))
        x = self.act(self.fn5(x))
        x = self.fn6(x)

        return x