# -*-coding:utf-8-*-

"""
判断是否带眼镜

"""
import pandas as pd
import numpy as np
import copy

# def load_data():
#     with open('lenses.data', 'r') as f:
#         lines = f.readlines()
#         data = []
#         for line in lines:
#             items = line.strip().split('  ')
#             data.append(items[1:])


def load_data():
    data = pd.read_fwf('lenses.data', names=['id', 'age of the patient', 'spectacle prescription', 'astigmatic', 'tear production rate', 'label'])
    data.pop('id')
    return data


def calculate_entropy(data):
    labels = data['label']
    prob = labels.value_counts(normalize=True)
    entropy = -prob.mul(np.log2(prob))
    return entropy.sum()


def decision_tree(train_data):
    # 计算系统熵
    sys_entropy = calculate_entropy(train_data)
    # 选择当下数据集中最大信息增益的列
    max_gain = {'gain': -1, 'sub_data': {}, 'col_name': ''}
    for col_name in train_data.keys():
        if col_name is 'label':
            continue
        # 计算条件熵
        entropy = 0
        train_data_size = train_data.shape[0]
        group_data = train_data.groupby(col_name)
        sub_data = {}
        for key, item in group_data.indices.items():
            records = train_data.iloc[item].copy()
            # 去除该列
            records.pop(col_name)
            sub_data[key] = records
            entropy += item.shape[0] / float(train_data_size) * calculate_entropy(records)
        gain = sys_entropy - entropy
        if gain > max_gain['gain']:
            max_gain['gain'] = gain
            max_gain['sub_data'] = sub_data
            max_gain['col_name'] = col_name
    # 构建树
    sub_tree = copy.deepcopy(sub_data)
    for key, item in max_gain['sub_data'].items():
        group_label = item.groupby('label').groups
        if len(group_label) == 1:
            sub_tree[int(key)] = int(item.iat[0, -1])
        elif item.keys().size == 2:
            # 获取标记最大值
            max_label = 0
            max_label_name = ''
            for label_key, label_val in group_label.items():
                if label_val.size > max_label:
                    max_label = label_val.size
                    max_label_name = label_key
            sub_tree[int(key)] = int(max_label_name)
        else:
            sub_tree[key] = decision_tree(item)
    tree = {max_gain['col_name']: sub_tree}
    return tree


def test_decision_tree(tree, test_data):
    for key, item in tree.items():
        test_val = item[test_data[key]]
        if isinstance(test_val, int):
            return test_val
        else:
            return test_decision_tree(test_val, test_data)


train_data = load_data()
tree = decision_tree(train_data)
test_data = pd.Series(data=[3, 1, 2, 2], index=['age of the patient', 'spectacle prescription', 'astigmatic', 'tear production rate'])
c = test_decision_tree(tree, test_data)

b = 1

