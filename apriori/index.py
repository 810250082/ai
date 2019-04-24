# -*-coding:utf-8-*-


from numpy import *
from itertools import combinations


# a = [1, 2, 3, 4]
# b = [frozenset([i]) for i in a]
# b = frozenset([1, 2, 3])
# c = combinations(b, 2)
# e = 1
# for item in combinations(b, 2):
#     print(item)


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
    s = []
    support_dict = {}
    for key, val in counter.items():
        support = val / m
        if support >= min_support:
            s.append(frozenset([key]))
        support_dict[frozenset([key])] = support
    return s, support_dict


def next_L(pre_set):
    """
    将集合组合成项数加1的集合
    :param pre_set:
    :param k:
    :return:
    """
    pre_num = len(pre_set)
    pre_item_num = len(pre_set[0])
    if len(pre_set) == 1:
        return []
    # 组合
    i = 0
    j = 1
    box = set([])
    while i < pre_num - 1:
        while j < pre_num:
            s = pre_set[i].union(pre_set[j])
            if len(s) == pre_item_num+1:
                box.add(s)
            j += 1
        i += 1
        j = i + 1

    return box


def filter_item(data, item_s, min_support=0.5):
    counter = {}
    for record in data:
        for item in item_s:
            if item.issubset(set(record)):
                if item not in counter:
                    counter[item] = 1
                else:
                    counter[item] += 1
    # 筛选频繁项集
    m = float(len(data))
    s = []
    support_dict = {}
    for key, val in counter.items():
        support = val / m
        if support >= min_support:
            s.append(key)
        support_dict[key] = support
    return s, support_dict


def apriori(data, min_support=0.5):
    s, support_dict = get_L1(data, min_support)
    if not s:
        return []
    k = 2
    count = len(s)
    box = s[:]
    while k <= count:
        next_s = next_L(s)
        s, supports = filter_item(data, next_s, min_support)
        support_dict.update(supports)
        if not s:
            break
        box.extend(s)
        k += 1
    return box, support_dict


def gen_rule(data, support_dict, min_support=0.7):
    box = []
    for item in data:
        if len(item) == 1:
            continue
        neces = [[]]
        for k in range(1, len(item)):
            # 获取长度为k的所有排列组合
            coms = list(combinations(item, k))
            # 筛选以 nece_item 为子集的所有项
            coms_copy = coms[:]
            for com in coms:
                b = 0
                for nece in neces:
                    if set(nece).issubset(set(com)):
                        b = 1
                        break
                if b == 0:
                    coms_copy.remove(com)
            # 生成规则
            all_rule = []
            for com in coms_copy:
                arr = [1, 1, 1]
                arr[1] = frozenset(com)
                arr[0] = item.difference(arr[1])
                arr[2] = support_dict[item] * 1.0 / support_dict[arr[0]]
                all_rule.append(arr)
            # 筛选规则
            neces = []
            for rule in all_rule:
                # 判断可信度
                if rule[2] >= min_support:
                    box.append(rule)
                    neces.append(rule[1])
            k += 1
    return box


# data = load_data()
# box, support_dict = apriori(data, min_support=0.5)
# rules = gen_rule(box, support_dict, min_support=0.5)
# b = 1

if __name__ == '__main__':
    data = []
    with open('mushroom.dat', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.strip().split(' '))
    # data = datas[datas[0] == 2][:, 1:].tolist()
    datas = array(data)
    data = datas[datas[:, 0] == '2'][:, 1:].tolist()
    box, support_dict = apriori(data, min_support=0.8)
    rules = gen_rule(box, support_dict, min_support=0.8)
    print(rules)

