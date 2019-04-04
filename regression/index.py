# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

# 产生数据
def product_data(num=100):
    X = np.linspace(0, 10, num=num, dtype=float)
    rand_num = 5 * np.random.random_sample((num,))
    Y = 3 * X + 1 + rand_num
    bias = np.ones(num)
    data = np.stack((bias, X, Y)).T
    np.savetxt('data.txt', data, fmt='%.2f', delimiter=' ')

product_data()
# 展示数据散点图

train_data = np.loadtxt('data.txt', dtype=float, delimiter=' ')
train_X = train_data[:, :-1]
train_Y = train_data[:, -1]
x = train_X[:, -1].tolist()
y = train_Y.tolist()
# plt.scatter(x, y)
# plt.show()


# 拟合数据 用公式求出w
def fit_data(train_X, train_Y):
    X = np.mat(train_X)
    Y = np.mat(train_Y).T
    W = (X.T * X).I * X.T * Y
    return W
# 展示直线
W = fit_data(train_X, train_Y)
pre_y = train_X * W

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y)
ax.plot(x, pre_y.T.A[0].tolist(), color='red')
plt.show()



