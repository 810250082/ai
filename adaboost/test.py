# -*-coding:utf-8-*-
from numpy import *


def load_data():
    data = [
        [0.1, 0.2],
        [1.5, 1.5],
        [0.3, 0.1],
        [1.1, 1.6],
        [0.2, 0.1],
        [0.5, 0.1],
        [1.1, 1.2],
        [1.2, 1.3],
        [1.3, 1.4],
        [0.4, 0.3],
    ]

    label = [-1, 1, -1, 1, -1, -1, 1, 1, 1, -1]

    return data, label


def calculate_rel(train_data, col_index, threshold, symbol):
    """
    根据选定的阀值计算结果
    :param train_data:
    :param col_index:
    :param threshold:
    :param symbol:
    :return:
    """
    mat_data = array(train_data)
    m, n = mat_data.shape
    rel = -ones(m)
    col_data = mat_data[:, col_index]
    if symbol == 'lg':
        rel[where(col_data <= threshold)] = 1
    else:
        rel[where(col_data > threshold)] = 1
    return rel


train_data, label = load_data()
calculate_rel(train_data, 0, 1.2, 'lg')
