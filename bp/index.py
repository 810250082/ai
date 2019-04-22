# -*-coding:utf-8-*-
"""
BP算法
"""
import numpy as np


"""
2*3*2 架构
"""


def load_data():
    input = [2, 3]
    out = [0.1, 0.9]
    return input, out


def sigmod(x):
    return 1.0 / (1 + np.exp(-1 * x))


def bf(input, output, step=0.001, max_iter=1000):
    input.append(1)
    input = np.array(input)
    output = np.array(output)
    m_input = input.shape[1]
    m_out = output.shape[0]
    w1 = np.random.random((m_input, 3))
    w2 = np.random.random((4, m_out))

    for i in range(max_iter):
        h_net = np.dot(input, w1)
        h_out = sigmod(h_net)
        h_out = np.vstack((h_out, [1]))
        o_net = np.dot(h_out, w2)
        o_out = sigmod(o_net)
        offset = o_out - output
        error = 1.0 / 2 * np.dot(offset, offset)
        g = (output - o_out) * o_out * (1 - o_out)
        w2 = np.dot(h_out, g)
        # w1 = g1 *
