import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001


class Net(torch.nn.Module):  # 开始搭建一个神经网络
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

dataMat = loadmat('datasets/MNIST.mat')
X_data = dataMat['X']
Y_data = dataMat['Y']
(n_samples, n_features), n_digits = X_data.shape, np.unique(Y_data).size
X_tensor = torch.tensor(X_data, dtype=torch.long)
Y_tensor = torch.tensor(Y_data, dtype=torch.long)
train_size = int(n_samples*0.8)
X_train = X_tensor[:train_size,:]
Y_train = Y_tensor[:train_size,:]
X_test = X_tensor[train_size:,:]
Y_test = Y_tensor[train_size:,:]

deal_train_dataset = TensorDataset(X_train, Y_train)
deal_test_dataset = TensorDataset(X_test, Y_test)

train_iter = DataLoader(deal_train_dataset, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(deal_test_dataset, batch_size=batch_size, shuffle=False)


net = Net(784,500,10).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# 训练网络
total_batch = len(train_iter)
for epoch in range(num_epochs):
    avg_cost = 0.0


    for i, (X,Y) in enumerate(train_iter):
        X = X.view(-1, 28*28).to(device)
        Y = Y.to(device)

        # 前向传播
        outputs = net(X)
        loss = criterion(outputs, Y)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 100 ==0:
            print('Epoch[{}/{}], Step[{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, i+1, total_batch, loss.item()))

