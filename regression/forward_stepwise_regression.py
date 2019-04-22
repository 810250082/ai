# -*-coding:utf-8-*-
"""
向前逐步回归
预测鲍鱼年龄
"""

from numpy import *


def load_data():
    datas = loadtxt('abalone.txt', delimiter='\t')
    data = datas[:, :-1]
    label = datas[:, -1]
    # 标准化数据
    data = data - mean(data, axis=0)
    data = data / std(data, axis=0)
    label = label - mean(label)
    return data, expand_dims(label, axis=1)


def calculer_error(data, label, w):
    mat_data = mat(data)
    label_data = mat(label)
    error = label_data - mat_data * w
    return error.T * error


def calculer_w(data, label, step=0.001, max_iter=5000):
    data = mat(data)
    label = mat(label)
    m, n = data.shape
    w = ones((n, 1))
    min_error = inf
    box = zeros((max_iter, n))
    for i in range(max_iter):
        for j in range(n):
            w_mid = w.copy()
            for symbol in [-1, 1]:
                w_new = w_mid.copy()
                # 产生一个新的w
                w_new[j][0] += symbol * step
                error = calculer_error(data, label, w_new)
                if error < min_error:
                    min_error = error
                    w = w_new.copy()
        box[i, :] = squeeze(w)
    return box


def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat


def stageWise(xArr,yArr,eps=0.001,numIt=5000):
    xMat = mat(xArr); yMat=mat(yArr)
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print ws.T
        lowestError = inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat


data, label = load_data()
# box = calculer_w(data, label)
box = stageWise(data, label)
print(box)
b = 1


