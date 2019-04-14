# -*-coding:utf-8-*-
"""
用梯度的下降法, 来推断马是否能够存活
1 整理数据
    将缺省的值置为0
    生成训练数据, 测试数据
2 生成器
    生成指定数一批的数据

3 sigmod 函数

4 利用梯度上升法, 更新 w
    初始化w, a
    迭代制定轮数
        根据seed打算数据
        创建生成器
        循环生成器
            从生成器获取data, label
            error
            W:=w - a * (error)*x
    返回w
4 测试数据
    大于0.5的为1, 小于0.5的为0
    统计错误率
"""

from numpy import *


def load_data(file_path):
    with open(file_path, 'r') as f:
        datas = f.readlines()
        arr = []
        for data_str in datas:
            arr.append(data_str.strip().split(' '))
        arr = array(arr)
        # 去除多余的列
        arr = delete(arr, [2, 23, 24, 25, 26, 27], 1)
        # 缺省的值置为0
        arr[arr =='?'] = 0
        arr = arr.astype(float)
        label = arr[:, -1].copy()
        label[label != 1] = 0
        arr[:, -1] = 1
    return arr, label


def sigmod(data, w):
    return 1.0 / (1 + exp(-1 * data * w))


def gen_datas(data, label, batch):
    m, n = data.shape
    i = 0
    while i <= m - batch:
        yield mat(data[i:i+batch, :]), mat(label[i:i+batch]).transpose()
        i += batch


def train(datas, labels, cycles=500, batch=5):
    m, n = datas.shape
    w = mat(ones((n, 1)))
    a = 0.001
    for cycle in range(cycles):
        # 打乱数据
        sed = random.randint(1, 100)
        ind = arange(m)
        random.seed(sed)
        random.shuffle(ind)
        shu_datas, shu_labels = datas[ind], labels[ind]
        # 创建生成器
        data_gen = gen_datas(shu_datas, shu_labels, batch)
        for data, label in data_gen:
            error = label - sigmod(data, w)
            w = w + a * data.transpose() * error
            # print(test(test_data, test_label, w))
    return w


def test(data, label, w):
    m, n = data.shape
    label = mat(label).transpose()
    pre_label = sigmod(data, w)
    pre_label[pre_label >= 0.5] = 1
    pre_label[pre_label < 0.5] = 0
    return sum(pre_label == label) * 1. / m


train_data, train_label = load_data('/home/ubuntu/protect/ai/LG/hourse/horse-colic.data')
test_data, test_label = load_data('/home/ubuntu/protect/ai/LG/hourse/horse-colic.test')

w = train(train_data, train_label)
rel = test(test_data, test_label, w)
print(rel)
