# -*- coding: utf-8 -*-
"""net_run.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CCS9hTo9gDAtoIsxHN2QFyVti9s5DKij

# Prepare
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import csv

"""# Import data"""

# train_dataset = pd.read_csv(
#     "/content/3_train.csv", delimiter=",")

# test_dataset = pd.read_csv(
#     "/content/3_test.csv", delimiter=",")

train_dataset = pd.read_csv(
    "test_train/1_train.csv", delimiter=",")

test_dataset = pd.read_csv(
    "test_train/1_test.csv", delimiter=",")

X_train = train_dataset['d']
X_train = np.asarray(X_train)
X_train = torch.from_numpy(X_train).type(torch.float).unsqueeze(1)
y_train = train_dataset['w']
y_train = np.asarray(y_train)
y_train = torch.from_numpy(y_train).type(torch.float).unsqueeze(1)

X_test = test_dataset['d']
X_test = np.asarray(X_test)
X_test = torch.from_numpy(X_test).type(torch.float).unsqueeze(1)
y_test = test_dataset['w']
y_test = np.asarray(y_test)
y_test = torch.from_numpy(y_test).type(torch.float).unsqueeze(1)

model_fine_x = pd.read_csv(
    "test_train/model_x.csv", delimiter=",")

X_fine = model_fine_x['d']
X_fine = np.asarray(X_fine)
X_fine = torch.from_numpy(X_fine).type(torch.float).unsqueeze(1)

"""# Construct nn"""


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_hidden),  # n_feature x n_hidden
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_hidden),  # n_hidden x n_hidden
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_output)  # n_hidden x n_output -> 1x1
        )

    def forward(self, x):
        return self.layers(x)

# nn parameter


epoches = 1001
models = 10
net = Net(n_feature=1, n_hidden=1000, n_output=1)
# print(net)
optimizer = torch.optim.SGD(
    net.parameters(), lr=0.08, momentum=0.5, weight_decay=0)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

"""# Train NN"""

# store result of training data set
y_prediction_train_list = [[[]]]

# plt.ion()
for i in range(models):
    for t in range(epoches):
        y_prediction_train = net(X_train)

        loss = loss_func(y_prediction_train, y_train)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if t == 1000:
            y_prediction_train = y_prediction_train.detach().numpy()
            y_prediction_train_list.append(y_prediction_train)

        # if t % 100 == 0:
        #     plt.cla()
        #     plt.scatter(X_train.data.numpy(), y_train.data.numpy())
        #     plt.plot(X_train.data.numpy(),
        #              y_prediction_train.data.numpy(), 'r-', lw=5)
        #     plt.xlabel("d")
        #     plt.ylabel("w")
        #     plt.pause(0.1)

    # plt.ioff()
    # plt.show()

final_y_pred_train = []

y_prediction_train_list.remove(y_prediction_train_list[0])
y_prediction_train_list = np.asarray(y_prediction_train_list)
y_prediction_train_list = np.transpose(y_prediction_train_list)

for i in y_prediction_train_list:
    for j in i:
        final_y_pred_train.append(np.average(j))


plt.scatter(X_train.data.numpy(), y_train.data.numpy())
plt.plot(X_train.data.numpy(),
         final_y_pred_train, 'r-', lw=5)
plt.show()


"""# Use nn to predict test"""

y_pred_test = net(X_test)

plt.scatter(X_test.data.numpy(), y_test.data.numpy())
plt.plot(X_test.data.numpy(), y_pred_test.data.numpy(), 'r-', lw=5)
plt.show()

y_fine = net(X_fine)

"""# Output data"""

with open('1_test_predict.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(y_pred_test)

with open('1_train_predict.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(final_y_pred_train)

with open('1_fine_predict.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerow(y_fine)