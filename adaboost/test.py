# -*-coding:utf-8-*-
from numpy import *


def load_data():
    data = [
        [0.1, 0.2],
        [1.5, 1.5],
        [0.3, 0.1],
        [1.1, 1.6],
        [0.2, 0.1],
        [0.5, 0.1],
        [1.1, 1.2],
        [1.2, 1.3],
        [1.3, 1.4],
        [0.4, 0.3],
    ]

    label = [-1, 1, -1, 1, -1, -1, 1, 1, 1, -1]

    return data, label


def calculate_rel(train_data, col_index, threshold, symbol):
    """
    根据选定的阀值计算结果
    :param train_data:
    :param col_index:
    :param threshold:
    :param symbol:
    :return:
    """
    mat_data = array(train_data)
    m, n = mat_data.shape
    rel = -ones(m)
    col_data = mat_data[:, col_index]
    if symbol == 'lt':
        rel[where(col_data <= threshold)] = 1
    else:
        rel[where(col_data > threshold)] = 1
    return rel


def optimum_stump(train_data, label_data, D):
    """
    获得在给定权重分布下最优树桩
    :return:
    """
    def get_threshold(col_data):
        # 给定列, 产生不同的阀值
        # 求出平均间隔, 然后再求出不同元素间的间隔
        max_val = max(col_data)
        min_val = min(col_data)
        num = col_data.shape[0]
        sort_val = sort(col_data)
        mean_interval = (max_val - min_val) / float(num)
        yield max_val + mean_interval
        yield min_val - mean_interval
        for key, item in enumerate(sort_val):
            if key > num - 2:
                break
            yield (sort_val[key] + sort_val[key + 1]) / 2.0

    min_threshold = 0
    min_col_index = 0
    min_error = inf
    min_symbol = 'lt'
    min_rel = None

    train_data = array(train_data)
    label_data = array(label_data)
    D = array(D)
    m, n = train_data.shape
    for col_index in range(n):
        gen_threshold = get_threshold(train_data[:, col_index])
        for threshold in gen_threshold:
            for symbol in ['lt', 'gt']:
                rel = calculate_rel(train_data, col_index, threshold, symbol)
                # 计算错误率
                error_indexs = where(label_data != rel)
                error = sum(D[error_indexs])

                if error < min_error:
                    min_threshold = threshold
                    min_col_index = col_index
                    min_error = error
                    min_symbol = symbol
                    min_rel = rel

    return min_threshold, min_col_index, min_error, min_symbol, min_rel


def adabost(train_data, label_data):
    """

    :param train_data:
    :param label_data:
    :return:
    """
    train_data = array(train_data)
    label_data = array(label_data)
    m, n = train_data.shape
    D = array([1.0 / m] * m)
    max_cycle = 40
    fx = []
    fx_rel = 0
    for i in range(max_cycle):
        # 产生树桩
        gx = {}
        threshold, col_index, error, symbol, rel = optimum_stump(train_data, label_data, D)
        alpha = 1.0 / 2 * log((1 - error) / (error + 0.00000001))
        gx['alpha'] = alpha
        gx['threshold'] = threshold
        gx['col_index'] = col_index
        gx['symbol'] = symbol
        fx.append(gx)
        D_new = D * exp(-alpha * label_data * rel)
        D /= sum(D_new)
        fx_rel += alpha * rel
        rel_sign = sign(fx_rel)
        if not sum(rel_sign != label_data):
            break
    return fx


def test(test_data, fx):
    test_data = array(test_data)
    test_fx = 0
    for g in fx:
        rel = calculate_rel(test_data, g['col_index'], g['threshold'], g['symbol'])
        test_fx += g['alpha'] * rel
    return sign(test_fx)


train_data, label = load_data()
fx = adabost(train_data, label)

test_data = [[0.1, 0.1], [1., 2.]]
test_ret = test(test_data, fx)
print(test_ret)