# -*-coding:utf-8-*-

# 准备数据
# 随机选择指定数量质心
# 循环聚簇,重置质心, 返回最后稳定的质心
# 展示数据,把簇圈起来

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(5)


def show_plt(data, centers):
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(centers[:, 0], centers[:, 1], color='red')
    plt.xlim(-1, 11)
    plt.ylim(-1, 11)
    plt.show()


# 准备数据
def product_data(section_X=[0, 1], section_Y=[0, 1], num=30):
    X = section_X[0] + (section_X[-1] - section_X[0]) * np.random.rand(num)
    Y = section_Y[0] + (section_Y[-1] - section_Y[0]) * np.random.rand(num)
    data = np.stack((X, Y)).T
    return data


points_1 = product_data(section_X=[0, 4], section_Y=[0, 4])
points_2 = product_data(section_X=[2, 6], section_Y=[6, 10])
points_3 = product_data(section_X=[6, 10], section_Y=[1, 5])
data = np.vstack((points_1, points_2, points_3))


# 随机选择指定数量质心
def select_centers(data=data, n=3):
    np.random.seed(None)
    min_x, min_y = np.min(data, axis=0)
    max_x, max_y = np.max(data, axis=0)
    centers = []
    while n:
        center_x = min_x + (max_x - min_x) * np.random.random()
        center_y = min_y + (max_y - min_y) * np.random.random()
        if (center_x, center_y) in data:
            continue

        centers.append((center_x, center_y))
        n -= 1
    return np.array(centers)


centers = select_centers(data, n=6)
show_plt(data, centers)


# 求欧式距离
def euc_distance(X, Y):
    return np.sqrt(np.sum(np.square(X - Y)))


# 循环聚簇,重置质心, 返回最后稳定的质心
def cluster_arith(data, centers, distance=euc_distance):
    cluster = -np.ones(data.shape)
    is_cycle = True
    while is_cycle:
        is_cycle = False
        # 圈点
        for i, item in enumerate(data):
            min_ind = -1
            min_distance = -np.inf
            # 求每个样本点到 那个质心的最小距离, 写入簇中
            for cen_ind, center in enumerate(centers):
                # 求欧式距离
                dis = distance(item, center)
                if min_ind == -1 or dis < min_distance:
                    min_ind = cen_ind
                    min_distance = dis
            if min_ind != cluster[i][0]:
                is_cycle = True
                cluster[i] = np.array([min_ind, min_distance])
        # 求新的质心
        for cen_ind, center in enumerate(centers):
            centers[cen_ind] = np.mean(data[cluster[:, 0] == cen_ind], axis=0)

        show_plt(data, centers)

    return centers


cent = cluster_arith(data, centers)
b = 1






# plt.show()





