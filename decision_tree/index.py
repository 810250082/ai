# -*-coding:utf-8-*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

# 定义一些常量
iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature_C = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
path = '/home/ubuntu/study/courseware/决策树/59d33a2a-0dd5-46eb-84b1-9a4fa3de84e7/datas/iris.data'
name = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
data = pd.read_csv(path, header=None, sep=',', names=name)

X = data[name[:-1]]
Y = data[name[-1]]
Y = pd.Categorical(Y).codes
# a = data.info()
# b = data.describe()

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=28)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=2)
tree.fit(x_train, y_train)

print("训练集上的准确率: {}".format(tree.score(x_train, y_train)))
print("测试集上的准确率: {}".format(tree.score(x_test, y_test)))

y_hat = tree.predict(x_test)
print('测试集上的准确率: {}'.format(np.mean(y_hat == y_test)))

b = 1

ss = MinMaxScaler()
ss.fit_transform(x_train)