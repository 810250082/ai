# -*-coding:utf-8-*-


import numpy as np
import matplotlib.pyplot as plt


def load_data():
    data = np.array([
        [0.1, 0.1, -1],
        [0.2, 0.1, -1],
        [0.3, 0.2, -1],
        [0.2, 0.4, -1],
        [0.3, 0.4, -1],
        [1.3, 1.0, 1],
        [1.2, 0.9, 1],
        [1.1, 1.4, 1],
        [1.3, 1.6, 1],
        [1.5, 1.4, 1],
    ])
    return data


def classify(data, test, k=3):
    """
    K紧邻判别方法
    :param data:
    :param test:
    :param k:
    :return:
    """
    distance = np.sum(np.square(data[:, :-1] - test), axis=1)
    k_orders = np.argsort(distance)[:k]
    result = np.sum(data[k_orders, -1])
    if result > 0:
        return 1
    else:
        return -1


data = load_data()
test = [0.8, 0.5]
rel = classify(data, test, k=5)
print(rel)

# 可视化
negative = data[data[:, -1] == -1][:-1]
positive = data[data[:, -1] == 1][:-1]
plt.scatter(negative[:, 0], negative[:, 1])
plt.scatter(positive[:, 0], positive[:, 1])
plt.scatter(test[0], test[1])
plt.show()
