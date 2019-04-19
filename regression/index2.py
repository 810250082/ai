from numpy import *
import matplotlib.pyplot as plt
#
#
# def load_data():
#     with open('data.txt', 'r') as f:
#         lines = f.readlines()
#         data = []
#         label = []
#         for line in lines:
#             arr = line.strip().split(' ')
#             data.append(arr)
#             label.append(1)
#     return data
#
#
# data = load_data()
# x = mat(data)
# b = 1


def load_data(filename):
    datas = loadtxt(filename, delimiter='\t')
    data = datas[:, :-1]
    label = datas[:, -1]
    return data, label


# def show_plot(data, label):
#     inds = data[:, 1].argsort(0)
#     x = data[inds][:, 1]
#     y = label[inds]
#     plt.scatter(x, y)


def lwlr(test_point, x_arr, y_arr, k=1.0):
    x_mat = mat(x_arr)
    y_mat = mat(y_arr).T
    m = shape(x_mat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = exp(diff_mat * diff_mat.T/(-2.0 * k**2))
    x_Tx = x_mat.T * (weights * x_mat)
    if linalg.det(x_Tx) == 0.0:
        print('cannot do inverse')
        return
    ws = x_Tx.I * (x_mat.T * (weights * y_mat))
    return test_point * ws


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    m = shape(test_arr)[0]
    y_hat = zeros(m)
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


data, label = load_data('ex0.txt')
# w = lwlr(data[0], data, label, 0.002)
y_hat = lwlr_test(data, data, label, 0.02)

ind = data[:, 1].argsort(0)
x_sort = data[ind]
y_sort = label[ind]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_sort[:, 1], y_hat[ind])

ax.scatter(x_sort[:, 1].flatten(), y_sort.flatten(), s=2, c='red')
plt.show()



# show_plot(data, label)
# inds = data[:, 1].argsort(0)
# x = data[inds][:, 1]
# y = label[inds]
# plt.scatter(x, y)
# # plt.show()
# plt.plot(data, y_hat)
# plt.show()
b = 1
