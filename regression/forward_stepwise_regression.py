# -*-coding:utf-8-*-
"""
向前逐步回归
预测鲍鱼年龄
"""

from numpy import *


def load_data():
    datas = loadtxt('abalone.txt', delimiter='\t')
    data = datas[:, -1]
    label = datas[-1]
    return data, expand_dims(label, axis=1)


def calculer_error(data, label, w):
    mat_data = mat(data)
    label_data = mat(label)
    error = label_data - mat_data * w
    return error.T * error


def calculer_w(data, label, step=0.001, max_iter=5000):
    data = mat(data)
    label = mat(label)
    m, n = data.shape
    w = ones((n, 1))
    min_error = inf
    for i in range(max_iter):
        for j in range(n):
            w_mid = w.copy()
            for symbol in [-1, 1]:
                w_new = w_mid.copy()
                # 产生一个新的w
                w_new = w_new[j][0] + symbol * step
                error = calculer_error(data, label, w_new)
                if error < min_error:
                    min_error = error
                    w = w_new.copy()
    return w


