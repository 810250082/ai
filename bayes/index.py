# -*-coding:utf-8-*-

"""
1
"""
import numpy as np

def product_data():
    data = [
        ['my', 'dog', 'has', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
    ]

    class_vec = [0, 1, 0, 1, 0, 1]
    return data, class_vec


def merge_datas(data):
    merge_data = set()
    for item in data:
        merge_data |= set(item)
    return list(merge_data)


def trans_to_vec(merge_data, data):
    zero_vec = [0] * len(merge_data)
    for item in data:
        if item in merge_data:
            zero_vec[merge_data.index(item)] = 1
    return zero_vec


def class_prob(data_vec, label_vec):
    label_len = len(label_vec)
    pb = float(sum(label_vec)) / label_len
    vec_len = len(data_vec[0])
    p0z = np.ones(vec_len)
    p1z = np.ones(vec_len)
    p0m = 2
    p1m = 2
    for i in range(label_len):
        if label_vec[i] == 0:
            p0z += np.array(data_vec[i])
            p0m += sum(data_vec[i])
        else:
            p1z += data_vec[i]
            p1m += sum(data_vec[i])
    p0 = np.log(p0z / p0m)
    p1 = np.log(p1z / p1m)

    return p0, p1, pb


def compare(test_vec, p0Vec, p1Vec, pb):
    test_vec = np.array(test_vec, dtype=float)
    p0 = np.log(pb) + np.sum(p0Vec * test_vec)
    p1 = np.log(pb) + np.sum(p1Vec * test_vec)

    if p0 > p1:
        return 0
    else:
        return 1


train_data, labels= product_data()
merge_data = merge_datas(train_data)
train_data_vecs = [trans_to_vec(merge_data, item) for item in train_data]

p0vec, p1vec, pb = class_prob(train_data_vecs, labels)

test_data = ['so', 'my', 'stupid', 'ate', 'garbage']
test_vec = trans_to_vec(merge_data, test_data)

rel = compare(test_vec, p0vec, p1vec, pb)
print(rel)
