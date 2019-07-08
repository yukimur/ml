
import numpy as np
from numpy.matrixlib import mat
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = [float(x) for x in line.strip().split('\t')]
        dataMat.append(lineArr[:-1])
        labelMat.append(lineArr[-1])
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx)==0.0:
        print("This matrix is singular,cannot do inverse")
        return
    ws = xTx.T*(xMat.T*yMat)
    return ws

def regression1(filename):
    xArr,yArr = loadDataSet(filename)
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres(xArr,yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten(),yMat[:,0].flatten().A[0])
    xCopy = xMat.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

if __name__ == "__main__":
    regression1('data.txt')