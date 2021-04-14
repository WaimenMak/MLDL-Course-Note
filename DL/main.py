# -*- coding: utf-8 -*-
# @Time    : 2021/4/10 18:08
# @Author  : Weiming Mai
# @FileName: main.py
# @Software: PyCharm

import numpy as np
from Nodes import *
from Utils import *
import xlrd

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
    #     min_max_scaler = preprocessing.MinMaxScaler()
    #     datamatrix = min_max_scaler.fit_transform(datamatrix)
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


batch_size = 16
x_batch, y_batch = mini_batch(batch_size, data_x,data_y)

# print(x_batch)
x = Input(x_batch)
# # l1.require_grad = False
l2 = Dense(x, 6)
# l2 = Sigmoid(l2)
l3 = Dense(l2, 4)
# l3 = Sigmoid(l3)
l4 = Dense(l3, 3)
# l4 = Sigmoid(l4)


y = Const(y_batch)
cost = MSE(l4, y)

Forward(cost)
p = Backprop(cost)
GD = G_D_Optimizer(p, 0.001, 10)
print(cost.value)
epoch = 160
for ep in range(epoch):
    x_batch, y_batch = mini_batch(batch_size, data_x,data_y)
    x.value = x_batch
    y.value = y_batch
    Forward(cost)
    Backprop(cost)
    GD.train()
    print(cost.value)
#
x.value = np.array([[5.1,3.4,1.5,0.2]])
Forward(l4)
# Backprop(cost)
print(l4.value)