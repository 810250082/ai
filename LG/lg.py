# -*-coding:utf-8-*-

import numpy as np

data_matrix = []
label_matrix = []

# 读数据
with open('datas', 'r') as f:
    arr = f.readlines()
    for item in arr:
        item_arr = item.strip().split()
        data_matrix.append([1.0, float(item_arr[0]), float(item_arr[1]), float(item_arr[2])])
        label_matrix.append([float(item_arr[3])])

print(data_matrix)
print(label_matrix)


def sigmod(Xint):
    return 1.0 / (1 + np.exp(-Xint))


def train():
    train_data = np.array(data_matrix)
    train_label = np.array(label_matrix)
    W = np.ones([train_data.shape[1], 1])
    r = 0.001
    n = 5000
    for i in range(n):
        error = train_label - sigmod(np.matmul(train_data, W))
        W = W + r * np.matmul(train_data.transpose(), error)
    return W

W = train()
x = np.array([[1.0, 93.85, 33.58, -75.27]])
# x = np.array([[1.0, -62.55, -78.23, -58.25]])
# x = np.array([[1.0, -95.56, 51.7, 3.27]])
rel = sigmod(np.matmul(x, W))

if rel > 0.5:
    print(1)
else:
    print(0)
