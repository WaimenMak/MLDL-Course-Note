# -*- coding: utf-8 -*-
# @Time    : 2022/4/9 11:18
# @Author  : Weiming Mai
# @FileName: model.py.py
# @Software: PyCharm

import torch
from torch.nn import functional as F
import torch.nn as nn
# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
def try_gpu(): #single gpu
    i = 0
    if torch.cuda.device_count() == i + 1:
        return torch.device(f'cuda:{i}')
    elif torch.cuda.device_count() == i + 2:
        return torch.device(f'cuda:{i+1}')
    return torch.device('cpu')


class block(nn.Module):
    def __init__(self, inpt_chann, oupt_chann, strides=1, conv1ks=False):
        super(block, self).__init__()
        self.component = nn.Sequential(
            nn.Conv2d(inpt_chann, oupt_chann, kernel_size=3, padding=1, stride=strides),
            nn.BatchNorm2d(oupt_chann),
            nn.ReLU(inplace=True),
            nn.Conv2d(oupt_chann, oupt_chann, kernel_size=3, padding=1),
            nn.BatchNorm2d(oupt_chann)
        )
        if conv1ks:  # shortcut convolution
            self.conv1ks = nn.Conv2d(inpt_chann, oupt_chann, kernel_size=1, stride=strides)
        else:
            self.conv1ks = None

    def forward(self, x):
        # self.component(x)
        if self.conv1ks:
            oupt = self.conv1ks(x) + self.component(x)
        else:
            oupt = x + self.component(x)

        return F.relu(oupt)


class resnet(nn.Module):
    def __init__(self, inpt_chann, classes):
        super(resnet, self).__init__()
        self.inpt = nn.Sequential(
            nn.Conv2d(inpt_chann, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.residual = nn.Sequential(self._make_layer(64, 64, 2, first_block=True),
                                      self._make_layer(64, 128, 2),
                                      self._make_layer(128, 256, 2),
                                      self._make_layer(256, 512, 2))

        self.oupt = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten(), nn.Linear(512, classes))

    def _make_layer(self, inpt_chann, oupt_chann, block_nums, first_block=False):
        blk = []
        for i in range(block_nums):
            if i == 0 and not first_block:
                blk.append(block(inpt_chann, oupt_chann, strides=2, conv1ks=True))
            else:
                blk.append(block(oupt_chann, oupt_chann))
        return nn.Sequential(*blk)

    def forward(self, x):
        h1 = self.inpt(x)
        res = self.residual(h1)
        fc = self.oupt(res)
        return fc


if __name__ == "__main__":
    net = resnet(3, 10)
    print(net)