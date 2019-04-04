# -*-coding:utf-8-*-
"""
id3算法实现
"""

#

"""
# 数据准备
#     使用花的真实数据
#     
# 模型构建
#     
# 输入模型
# 产生树
# 预测数据

决策树构建过程
计算各个属性的信息增益
    
选择最大的划分数据
同样的算法递归到子树
"""
import pandas as pd

# datas = pd.read_excel('datas.xlsx')
datas = pd.read_excel('datas.xlsx', names=['id', 'has_hours', 'merry', 'income', 'no_pay'])
datas.pop('id')


# class node(object):
#     """
#     节点
#     """
#     prev =

def build_decision_tree(datas):
    """
    构建决策数
    :return:
    """
    def gain():
        """
        获取信息增益
        :return:
        """
        pass

    col_gain = {}
    max_gain_name = ''
    for item in datas.columns:
        col_gain[item], example_groups = gain(datas, item)
        if not max_gain_name:
            max_gain_name = item
        if col_gain[item] > col_gain[max_gain_name]:
            max_gain_name = item



b = 1
