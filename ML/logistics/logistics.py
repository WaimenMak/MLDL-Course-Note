import numpy as np
import pandas as pd
from sklearn import preprocessing

# X = pd.read_csv('X_train')
# X = X.iloc[:,1:].to_numpy()
# Y = pd.read_csv('Y_train')
# Y = Y.iloc[:,1:].to_numpy()
# MinMax = preprocessing.MinMaxScaler()
# x_scale = MinMax.fit_transform(X)
# data_x = x_scale
# data_y = Y

### for SGD
def sigmoid(x):
    y = 1/(1+np.exp(x))
    return y


def mini_batch(batch_size, data_X, data_Y):
    idx = np.arange(0, data_X.shape[0])
    np.random.shuffle(idx)
    idx = idx[0:batch_size]

    return data_X[idx,:], data_Y[idx,:]

class logistics_regression():

    def __init__(self,X_train, label):
        self.X = X_train
        self.Y = label
        self.weight1 = np.random.normal(0,0.5,[1,self.X.shape[1]])
        # self.weight2 = np.random.normal(0,0.5,[1,self.X.shape[1]])
        self.b = np.random.normal(0,0.5,[1,1])
        self.lr = 0.001
        # self.g_w = np.zeros(self.weight1.shape)
        # self.g_b = np.zeros(self.b.shape)
        self.loss = 0
        self.eps = 0.00001

    def train(self):
        m = self.X.shape[0]
        loss = 0

        # sum_gw2 = 0

        for i in range(m):
            for j in range(100):
                output = self.forward(self.X[i,:])
                gw = (output - self.Y[i,:])*self.X[i,:]
                self.weight1 = self.weight1 - self.lr * gw
                gb = (output - self.Y[i,:])
                self.b = self.b - self.lr * gb



        for j in range(m):
            self.loss += self.loss + self.Y[j,:]*np.log(self.forward(self.X[j,:]) + self.eps) + (1 - self.Y[j,:])*np.log(1 - self.forward(self.X[j,:]) + self.eps)   
        # self.loss = -self.loss

    def forward(self, x):
        y_pred = sigmoid(np.dot(self.weight1, x) + self.b)
        # y_pred = np.dot(self.weight1, x) + np.dot(self.weight2, x**2) + self.b
        return y_pred 

iter = 10
batch_size = 128

x = np.array([[1,1,1]])
y = np.array([[1]])
data_x = x
data_y = y

x_batch, y_batch = mini_batch(batch_size, data_x,data_y)
LG = logistics_regression(x_batch,y_batch)
for epoch in range(iter):
    x_batch, y_batch = mini_batch(batch_size, data_x,data_y)
    LG.X = x_batch
    LG.Y = y_batch
    for i in range(100):
        LG.loss = 0
        LG.train()
        if i % 100 == 0:
            print(LG.loss)