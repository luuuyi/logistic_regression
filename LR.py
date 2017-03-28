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

def plotBeatFit(weight,file_name):
    import matplotlib.pyplot as plt
    wet_array = weight.getA()
    datas, labels = loadDatasFromFile(file_name)
    datas = array(datas)
    height = shape(datas)[0]
    xcord1 = [];ycord1 = []; 
    xcord2 = [];ycord2 = []; 
    for i in range(height):
        if int(labels[i]) == 1:
            xcord1.append(datas[i][1]); ycord1.append(datas[i][2])
        else:
            xcord2.append(datas[i][1]); ycord2.append(datas[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-wet_array[0]-wet_array[1]*x)/wet_array[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2'); 
    plt.show()

if __name__ == '__main__':
    file_name = 'testSet.txt'
    datas, labels = loadDatasFromFile(file_name)
    #print datas, labels
    weights = gradientAscent(datas,labels)
    #print weights
    plotBeatFit(weights,file_name)