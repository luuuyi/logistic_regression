from numpy import *
import sys

def loadDatasFromFile(file):
    fd = open(file)
    lines = fd.readlines()
    datas = []
    labels = []
    for line in lines:
        tmp_list = line.strip().split('\t')
        datas.append([1.0,float(tmp_list[0]),float(tmp_list[1])])
        labels.append(float(tmp_list[-1]))
    return datas, labels

def sigmoid(in_datas):
    return 1.0/(1.0+exp(-in_datas))

def gradientAscent(datas,labels):
    datas_mat = mat(datas)
    labels_mat = mat(labels).transpose()
    h, w = shape(datas_mat)
    alpha = 0.001
    max_iter = 500
    weights = ones((w,1))
    for i in range(max_iter):
        res = sigmoid(datas_mat*weights)
        errors = (labels_mat - res)
        weights = weights + alpha*datas_mat.transpose()*errors          #梯度上升变形，朝着标准标签的方向缓慢移动
    return weights


if __name__ == '__main__':
    datas, labels = loadDatasFromFile('testSet.txt')
    #print datas, labels
    weights = gradientAscent(datas,labels)
    print weights