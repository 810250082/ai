# -*-coding:utf-8-*-


from numpy import *


def load_data():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def get_L1(data, min_support=0.5):
    """
    获取单个频繁项集的集合
    :param data:
    :param min_support:
    :return:
    """
    # 统计支持度
    counter = {}
    for record in data:
        for item in record:
            if item not in counter:
                counter[item] = 1
            else:
                counter[item] += 1
    # 筛选频繁项集
    m = float(len(data))
    s = set([])
    for key, val in counter.items():
        if val / m >= min_support:
            s.add((key,))
    return s


def next_L(pre_set):
    """
    将集合组合成项数加1的集合
    :param pre_set:
    :param k:
    :return:
    """
    pre_num = len(pre_set)
    pre_set = list(pre_set)
    pre_item_num = len(list(pre_set)[0])
    if len(pre_set) == 1:
        return []
    # 组合
    i = 0
    j = 1
    box = set([])
    while i < pre_num - 1:
        while j < pre_num:
            s = set(pre_set[i]+pre_set[j])
            if len(s) == pre_item_num+1:
                box.add(tuple(s))
            j += 1
        i += 1
        j = i + 1

    return box


def filter_item(data, item_s, min_support=0.5):
    counter = {}
    for record in data:
        for item in item_s:
            if set(item).issubset(set(record)):
                if item not in counter:
                    counter[item] = 1
                else:
                    counter[item] += 1
    # 筛选频繁项集
    m = float(len(data))
    s = set([])
    for key, val in counter.items():
        if val / m >= min_support:
            s.add(key)
    return s


def apriori(data, min_support=0.5):
    s1 = get_L1(data, min_support)
    if not s1:
        return []
    k = 2
    box = s1
    while k <= len(s1):
        next_s = next_L(s1)
        filter_s = filter_item(data, next_s, min_support)
        if not filter_s:
            break
        box |= filter_s
        k += 1
    return box


data = load_data()
a = apriori(data, min_support=0.7)
b = 1

