# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 18:08
# @Author  : Weiming Mai
# @FileName: main.py
# @Software: PyCharm

import numpy as np
from Nodes import *
from Utils import *
import xlrd
from sklearn import preprocessing

# read data
def excel_to_matrix(path, num):
    table = xlrd.open_workbook(path).sheets()[num]  # 获取第一个sheet表
    row = table.nrows  # 行数
    col = table.ncols  # 列数
    datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col):
        cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols  # 按列把数据存进矩阵中
        # 数据归一化
        min_max_scaler = preprocessing.MinMaxScaler()
        datamatrix = min_max_scaler.fit_transform(datamatrix)
    return datamatrix

def mini_batch(batch_size, data_X, data_Y):
    idx = np.arange(0, data_X.shape[0])
    np.random.shuffle(idx)
    idx = idx[0:batch_size]

    return data_X[idx,:], data_Y[idx,:]

datafile = "C:/Users/13271/Desktop/git/DL/iris_test+train.xlsx"

data_x = excel_to_matrix(datafile, 0)
data_y = excel_to_matrix(datafile, 3)

x_test = excel_to_matrix(datafile, 1)
y_test = excel_to_matrix(datafile, 2)


batch_size = 64
x_batch, y_batch = mini_batch(batch_size, data_x, data_y)

x = Input(x_batch)
l2 = Dense(x, 6)
l2 = batch_normalize(l2)
# l2 = Sigmoid(l2)
# l3 = Dense(l2, 4)
# l3 = Sigmoid(l3)
# l4 = Dense(l3, 4)
# l4 = Sigmoid(l4)
l5 = Dense(l2, 3)
l5 = Sigmoid(l5)


y = Const(y_batch)
cost = MSE(l5, y)

Forward(cost)
p = Train(cost, [])
GD = G_D_Optimizer(p, 0.05) # lr
print(cost.value)
epoch = 1000
for ep in range(epoch):
    x_batch, y_batch = mini_batch(batch_size, data_x,data_y)
    x.value = x_batch
    y.value = y_batch
    for j in range(20):
        Forward(cost)
        Backprop(cost)
        GD.train()
    if ep % 100 == 0:
        print(cost.value)

Test(cost)
x.value = x_test
Forward(cost)
# Backprop(cost)
print(cost.value)
print(l5.value)