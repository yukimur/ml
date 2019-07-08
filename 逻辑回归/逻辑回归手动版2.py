
import numpy as np
from numpy.matrixlib import mat
import matplotlib.pyplot as plt


def loadDataSet(file_name):
    """加载数据"""
    dataMat = []
    labelMat = []
    fr = open(file_name)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def plotBestFit(dataArr,labelMat,weights):
    # xcord1 = []
    # xcord2 = []
    # ycord1 = []
    # ycord2 = []
    # for i in range(dataArr.shape[0]):
    #     if int(labelMat[i])== 1:
    #         xcord1.append(dataArr[i,1])
    #         ycord1.append(dataArr[i,2])
    #     else:
    #         xcord2.append(dataArr[i,1])
    #         ycord2.append(dataArr[i,2])
    labelMat = np.array(labelMat.transpose().tolist()[0])
    dataArr = np.array(dataArr)
    xcord1 = dataArr[labelMat == 1, 1]
    ycord1 = dataArr[labelMat == 1, 2]
    xcord2 = dataArr[labelMat == 0, 1]
    ycord2 = dataArr[labelMat == 0, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)

    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x,y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def sigmoid(inX):
    a = 1.0/(1+1.0/np.exp(inX))
    return a

def stocGradAscent2(dataMatrix,labelMat,numIter=150):
    m,n = dataMatrix.shape
    weights = np.ones((n,1))
    for j in range(numIter):
        for i in range(m):
            lr = 4 / (1.0 + j + i) + 0.0001
            h = sigmoid(dataMatrix[i]*weights)
            error = labelMat[i]-h
            weights = weights + lr*dataMatrix[i].transpose()*error
    return np.array(weights)

def stocGradAscent1(dataMatrix,labelMat):
    """随机梯度上升"""
    m,n = dataMatrix.shape
    lr = 0.001
    weights = np.ones((n,1))
    for i in range(m):
        h = sigmoid(dataMatrix[i]*weights)
        error = labelMat[i]-h
        weights = weights + lr*dataMatrix[i].transpose()*error
    return np.array(weights)

def stocGradAscent(dataMatrix,labelMat):
    m,n = dataMatrix.shape
    lr = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = labelMat-h
        weights = weights + lr*dataMatrix.transpose()*error
    return np.array(weights)

def ceshiLR(filename):
    # 加载数据
    dataMat,labelMat = loadDataSet(filename)
    # 转换数据
    dataArr = mat(dataMat)
    labelMat = mat(labelMat).transpose()
    weights = stocGradAscent2(dataArr,labelMat)
    plotBestFit(dataArr,labelMat,weights)

if __name__ == "__main__":
    ceshiLR('data.txt')