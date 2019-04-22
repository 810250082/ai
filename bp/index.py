# -*-coding:utf-8-*-
"""
BP算法
"""
from numpy import *


"""
2*3*2 架构
"""


def load_data():
    inputs = mat([2, 3])
    out = mat([0.1, 0.9])
    b = array([1, 1])
    return inputs, out, b


def sigmod(x):
    return 1.0 / (1 + exp(-1 * x))


def bf(inputs, output, b, step=0.001, max_iter=1000):
    m_input = inputs.shape[1]
    m_out = output.shape[1]
    w1 = random.random((m_input, 3))
    w2 = random.random((3, m_out))

    for i in range(max_iter):
        h_net = inputs * w1 + b[0]
        h_out = sigmod(h_net)
        o_net = h_out * w2 + b[1]
        o_out = sigmod(o_net)
        offset = o_out - output
        error = 1.0 / 2 * offset * offset.transpose()
        g = -(output.A - o_out.A) * o_out.A * (1 - o_out.A)
        w2 = w2 - step * h_out.transpose() * g
        g1 = g * w2.transpose()
        w1 = w1 - step * inputs.transpose() * g1
        b[1] = b[1] - step * sum(g)
        b[0] = b[0] - step * sum(g1)
        print('error: {}'.format(error))


inputs, out, b = load_data()
bf(inputs, out, b, step=0.001, max_iter=500000)