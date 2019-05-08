# -*-coding:utf-8-*-


from numpy import *


def load_data():
    data = loadtxt('testSet.txt', delimiter='\t')
    return data


def tran_matrix(data, k=None):
    data = mat(data)
    if k is None:
        k = data.shape[1]
    # 原始数据均值化
    data = data - mean(data, axis=1)
    # 求协方差矩阵
    C = data.T * data
    # 求C的特征向量, 特征根
    eig_val, feat_vect = linalg.eig(C)
    argsort(eig_val)