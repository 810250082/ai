# -*-coding:utf-8-*-
import numpy as np

from scipy.stats import multivariate_normal

def init_dataset():
    mean1 = (0, 0, 0)
    cov1 = np.diag((1, 1, 1))
    data1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=500)

    mean2 = (1, 1, 1)
    cov2 = np.diag((2, 2, 2))
    data2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=500)
    data = np.vstack((data1, data2))
    return data

data = init_dataset()

# mean = [0, 0]
# cov = [[1, 0], [0, 1]]  # diagonal covariance
#
#
# import matplotlib.pyplot as plt
# x, y = np.random.multivariate_normal(mean, cov, 5000).T
# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.show()

def train(x, max_iter=100):
    m, n = np.shape(x)
    mu1 = x.min(axis=0)
    mu2 = x.max(axis=0)

    sigma1 = np.identity(n)
    sigma2 = np.identity(n)

    pi = 0.5

    for i in range(max_iter):
        norm1 = multivariate_normal(mu1, sigma1)
        norm2 = multivariate_normal(mu2, sigma2)
        tau1 = pi * norm1.pdf(x)
        tau2 = (1 - pi) * norm2.pdf(x)
        w = tau1 / (tau1 + tau2)

        mu1 = np.dot(w, x) / np.sum(w)
        mu2 = np.dot((1 - w), x) / np.sum((1 - w))
        sigma1 = np.dot(w * (x-mu1).T, x-mu1) / np.sum(w)
        sigma2 = np.dot((1-w) * (x-mu2).T, x-mu2) / np.sum(1-w)
        pi = np.sum(w) / m

    return pi, mu1, mu2, sigma1, sigma2


if __name__ == '__main__':
    data = init_dataset()
    pi, mu1, mu2, sigma1, sigma2 = train(data, 100)

    x = np.array([0, -1, 0])
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    p1 = pi * norm1.pdf(x)
    p2 = (1-pi) * norm2.pdf(x)
    if p1 > p2:
        print('第一类别')
    else:
        print('第二类别')