import pandas as pd
import numpy  as np
from sklearn import preprocessing


class linear_regression():

    def __init__(self,X_train, label):
        self.X = X_train
        self.Y = label
        self.weight1 = np.random.normal(0,0.5,[1,self.X.shape[1]])
        # self.weight2 = np.random.normal(0,0.5,[1,self.X.shape[1]])
        self.b = np.random.normal(0,0.5,[1,1])
        self.lr = 0.01
        # self.g_w = np.zeros(self.weight1.shape)
        # self.g_b = np.zeros(self.b.shape)
        self.loss = 0

    def train(self):
        m = self.X.shape[0]
        loss = 0

        # sum_gw2 = 0

        for i in range(m):
            # sum_gw = 0
            # sum_gb = 0
            for j in range(10):           # each sample learn 10 times
                output = self.forward(self.X[i,:])
                gw = 2*(output - self.Y[i,:])*self.X[i,:]
                # sum_gw += gw**2
                # self.weight1 = self.weight1 - self.lr * gw / np.sqrt(sum_gw)
                self.weight1 = self.weight1 - self.lr * gw
                # sum_gw2 += 1/m*2*(output - self.Y[i,:])*self.X[i,:]**2
                gb = 2*(output - self.Y[i,:])
                # sum_gb += gb ** 2
                # self.b = self.b - self.lr * gb / np.sqrt(sum_gb)
                self.b = self.b - self.lr * gb

        # self.g_w += sum_gw**2
        # self.g_b += sum_gb**2
        # self.weight1 = self.weight1 - self.lr*sum_gw/np.sqrt(self.g_w)
        # self.weight2 = self.weight2 - self.lr*sum_gw2
        # self.b = self.b - self.lr*sum_gb/np.sqrt(self.g_b)

        for j in range(m):
            self.loss += self.loss + np.sqrt(1/m*(self.forward(self.X[j,:]) - self.Y[j,:])**2)

    def forward(self, x):
        y_pred = np.dot(self.weight1, x) + self.b
        # y_pred = np.dot(self.weight1, x) + np.dot(self.weight2, x**2) + self.b
        return y_pred 
    

def  sliding_window(data,width,X,Y):
    '''
    width = 9
    data = data_set
    '''
    for i in range(24*20*12-9-1):
        X[i,:] = data[i:i+width,:].reshape([1,width*18])
        Y[i,:] = data[i+width,9]
    return X,Y

def mini_batch(batch_size, data_X, data_Y):
    idx = np.arange(0, data_X.shape[0])
    np.random.shuffle(idx)
    idx = idx[0:batch_size]

    return data_X[idx,:], data_Y[idx,:]

data = pd.read_csv('train.csv', encoding = 'big5')
new_data = data.iloc[:,3:]
new_data[new_data == 'NR'] = 0
raw_data = new_data.to_numpy()

X = np.zeros([24*20*12-9-1,9*18])
Y = np.zeros([24*20*12-9-1,1])

data_set = np.zeros([24*20*12,18])
n = 0
for i in range(0,24*20*12,24):
    data_set[i:i+24,:] = raw_data[n:n+18,:].T

X,Y = sliding_window(data_set,9,X,Y)
Y.shape

iter = 20
batch_size = 50
MinMax = preprocessing.MinMaxScaler()
# x = np.array([[1],[2],[3]])
# y = np.array([[10],[20],[30]])
x_scale = MinMax.fit_transform(X)
datax = x_scale
datay = Y
x_batch, y_batch = mini_batch(batch_size, datax,datay)
LR = linear_regression(x_batch,y_batch)
for epoch in range(iter):
    x_batch, y_batch = mini_batch(batch_size, datax,datay)
    LR.X = x_batch
    LR.Y = y_batch
    for i in range(100):
        LR.loss = 0
        LR.train()
        if i % 20 == 0:
            print(LR.loss)

print(LR.forward(x_scale[0,:]))
print(Y[0,:])