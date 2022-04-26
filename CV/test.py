# -*- coding: utf-8 -*-
# @Time    : 2022/4/10 13:12
# @Author  : Weiming Mai
# @FileName: test.py
# @Software: PyCharm

import torch
import os
from torch.nn import functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import try_gpu, resnet
from Data import trainset, augment_loader, augment_loader2

if __name__ == "__main__":
    labels_dataframe = pd.read_csv('./classify-leaves/train.csv')
    leaves_labels = sorted(list(set(labels_dataframe['label'])))
    n_classes = len(leaves_labels)
    num_label_dict = dict(zip(leaves_labels, range(n_classes)))
    transform_to_str = dict()
    for i, key in enumerate(num_label_dict.keys()):
        transform_to_str[i] = key

    device = try_gpu()
    # device = "cuda:0"
    # net = resnet(3, n_classes)
    net = torch.load("./exp/resnet_preaug.pth", map_location='cuda:0')
    net.to(device)
    testset = trainset(train=False, test=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    ###### Test ######
    prediction = []
    net.eval()
    print("Begin test")
    for X in testloader:
        X = X.to(device)
        with torch.no_grad():
            pred = net(X)
            # acc = (pred.argmax(dim=-1) == Y).float().mean()
        prediction.extend(pred.argmax(dim=-1).cpu().numpy().tolist())          #each batch

    preds = []
    for pred in prediction:
        preds.append(transform_to_str[pred])

    submission = pd.read_csv('./classify-leaves/test.csv')
    submission["label"] = pd.Series(preds)
    # submission = pd.concat([submission['image'], submission['label']], axis=1)
    submission.to_csv("test_result_aug3.csv", index=False)
    print(f"Finish Test")

    #normal: 0.83977
    #aug1: 0.85295 flip
    #aug2:0.3150 erase
    #aug3: 0.652 centercrop

    #pre 0.940
    #pre aug:0.926