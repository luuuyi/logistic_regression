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

def randomGradientAscent(datas,labels):
    datas_arr = array(datas)            #array和matrix有区别
    h, w = shape(datas_arr)
    alpha = 0.01
    weights = ones(w)                   #一维矩阵
    for i in range(h):
        res = sigmoid(sum(datas_arr[i]*weights))
        errors = labels[i] - res
        weights = weights + alpha*errors*datas_arr[i]
    return weights

def advancedRandomGradientAscent(datas,labels,iter_nums=150):
    import random
    datas_arr = array(datas)           
    h, w = shape(datas_arr)
    weights = ones(w)                  
    for i in range(iter_nums):
        indexs = range(h)
        for j in range(h):
            alpha = 4/(1.0+j+i)+0.01
            random_index = int(random.uniform(0,len(indexs)))
            res = sigmoid(sum(datas_arr[random_index]*weights))
            errors = labels[random_index] - res
            weights = weights + alpha*errors*datas_arr[random_index]
            del(indexs[random_index])
    return weights

def classifyResult(datas,weights):
    pro = sigmoid(sum(datas*weights))
    if pro >= 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    fd_train = open('horseColicTraining.txt')
    fd_test = open('horseColicTest.txt')
    train_datas=[]; train_labels=[]
    for line in fd_train.readlines():
        line_data_list = line.strip().split('\t')
        tmp_list=[]
        for i in range(21):
            tmp_list.append(float(line_data_list[i]))
        train_datas.append(tmp_list)
        train_labels.append(float(line_data_list[21]))
    weights = advancedRandomGradientAscent(train_datas,train_labels,500)
    error_count = 0; total_size = 0.0
    for line in fd_test.readlines():
        total_size += 1.0
        line_data_list = line.strip().split('\t')
        test_data = []
        for i in range(21):
            test_data.append(float(line_data_list[i]))
        if int(classifyResult(array(test_data),weights)) != int(line_data_list[21]):
            error_count += 1
    error_rate = float(error_count)/total_size
    print "The Error Ratio is: %f" % error_rate
    return error_rate

def multiTest():
    nums = 10; errors = 0.0
    for i in range(nums):
        errors += colicTest()
    errors_mean = errors / 10.0
    print "The Mean Error Ratio is: %f" % errors_mean
    return errors_mean

def plotBeatFit(wet_array,file_name):
    import matplotlib.pyplot as plt
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
    #datas, labels = loadDatasFromFile(file_name)
    #weights = advancedRandomGradientAscent(datas,labels)
    #print datas, labels
    #weights = gradientAscent(datas,labels)
    #plotBeatFit(weights.getA(),file_name)
    #weights = randomGradientAscent(datas,labels)
    #print weights
    #plotBeatFit(weights,file_name)
    #colicTest()
    multiTest()