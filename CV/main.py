# -*- coding: utf-8 -*-
# @Time    : 2022/4/9 11:18
# @Author  : Weiming Mai
# @FileName: main.py.py
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
import torchvision.models as models

net = models.resnet18(pretrained=True)
if __name__ == "__main__":
    torch.manual_seed = 1
    aug = True
    model_path = "./model_path"
    if aug:
        train_data = trainset(loader=augment_loader)
        # train_data = trainset(loader=augment_loader)
        trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    else:
        train_data = trainset()
        trainloader = DataLoader(train_data, batch_size=64, shuffle=True)

    val_data = trainset(train=False, val=True)
    valloader = DataLoader(val_data, batch_size=64, shuffle=True)

    learning_rate = 3e-4
    wd = 1e-3
    n_classes = 176

    epochs = 50
    device = try_gpu()
    # net = resnet(3, n_classes)
    net = models.resnet18(pretrained=True)  #pretrain
    net.to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wd)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    epoch_loss = []
    mean_acc = []
    epoch_train_acc = []
    print("Begin Train:")

    for epc in range(epochs):
        train_loss = []
        val_acc = []
        train_acc = []
        #### train #####
        net.train()
        for X, Y in trainloader:
            X, Y = X.to(device), Y.to(device)
            with torch.no_grad():
                train_pred = net(X)
                tr_acc = (train_pred.argmax(dim=-1) == Y).float().mean()
            train_acc.append(tr_acc.item())
            oupt = net(X)
            optimizer.zero_grad()
            loss = loss_fn(oupt, Y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data)

        trainacc = sum(train_acc) / len(train_acc)
        epoch_train_acc.append(trainacc)
        loss_info = sum(train_loss) / len(train_loss)
        epoch_loss.append(loss_info.cpu())
        print(f"Training Loss:{loss_info:.2f}", end="//")
        print(f"Training Acc:{trainacc:.2f}, Epoch:{epc}/{epochs}", end="//")

        #### val #####
        net.eval()
        for X, Y in valloader:
            X, Y = X.to(device), Y.to(device)
            with torch.no_grad():
                pred = net(X)
                acc = (pred.argmax(dim=-1) == Y).float().mean()
            val_acc.append(acc.item())          #each batch

        acc_info = sum(val_acc) / len(val_acc)    #each epoch
        mean_acc.append(acc_info)
        print(f"Validation Acc:{acc_info:.2f}, Epoch:{epc}/{epochs}")

        # np.save(model_path + "/val_acc_normal.npy", mean_acc)
        # np.save(model_path + "/train_loss_normal.npy", epoch_loss)

    # np.save("val_acc_normal", val_acc)
    # np.save("train_loss_normal", train_loss)

    if not aug:
        try:
              # 文件保存路径，如果不存在就会被重建
            if not os.path.exists(model_path):  # 如果路径不存在
                os.makedirs(model_path)
            np.save(model_path + "/val_acc_pre.npy", mean_acc)
            np.save(model_path + "/train_acc_pre.npy", epoch_train_acc)
            np.save(model_path + "/train_loss_pre.npy", epoch_loss)
            torch.save(net, model_path+"/resnet.pth")
        except:
            np.save(model_path + "/val_acc_pre.npy", mean_acc)
            np.save(model_path + "/train_acc_pre.npy", epoch_train_acc)
            np.save(model_path + "/train_loss_pre.npy", epoch_loss)
            torch.save(net, "/resnet.pth")
    else:
        if not os.path.exists(model_path):  # 如果路径不存在
            os.makedirs(model_path)
        np.save(model_path+"/val_acc_preaug.npy", mean_acc)
        np.save(model_path + "/train_acc_preaug.npy", epoch_train_acc)
        np.save(model_path+"/train_loss_preaug.npy", epoch_loss)
        torch.save(net, model_path+"/resnet_preaug.pth")