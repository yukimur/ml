
import numpy as np
from matplotlib import pyplot as plt

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

def sigmoid(inX):
    return 1.0/(1+1/np.exp(inX))

def gradAscent(dataMatrix,labelMatrix):
    m,n = np.shape(dataMatrix)  # 样本数量,特征数量
    lr = 0.001
    maxCycles = 100
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(np.dot(dataMatrix,weights))
        error = labelMatrix-h
        weights = weights + lr*np.dot(dataMatrix.transpose(),error)
    return weights

def stocGradAscent1(dataMatrix,labelMatrix):
    """随机梯度上升"""
    m,n = np.shape(dataMatrix)  # 样本数量,特征数量
    lr = 0.001

    weights = np.ones((n,1))
    for k in range(m):
        h = sigmoid(np.dot(dataMatrix[k:k+1,:],weights))
        error = labelMatrix[k]-h
        weights = weights + lr*np.dot(dataMatrix[k:k+1,:].transpose(),error)
    return weights

def stocGradAscent2(dataMatrix,labelMatrix,numIter=150):
    """改进的随机梯度下降"""
    m,n = np.shape(dataMatrix)  # 样本数量,特征数量

    weights = np.ones((n,1))
    for i in range(numIter):
        for k in range(m):
            lr = 4 / (1.0 + k + i) + 0.0001
            h = sigmoid(np.dot(dataMatrix[k:k+1,:],weights))
            error = labelMatrix[k]-h
            weights = weights + lr*np.dot(dataMatrix[k:k+1,:].transpose(),error)
    return weights

def plotBestFit(dataMatrix,labelMatrix,weights):
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(dataMatrix.shape[0]):
        if int(labelMatrix[i])== 1:
            xcord1.append(dataMatrix[i,1])
            ycord1.append(dataMatrix[i,2])
        else:
            xcord2.append(dataMatrix[i,1])
            ycord2.append(dataMatrix[i,2])

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

def ceshiLR():
    dataMat,labelMat = loadDataSet("data.txt")
    dataMatrix = np.array(dataMat)
    labelMatrix = np.array(labelMat).reshape((dataMatrix.shape[0],1))
    weights = stocGradAscent2(dataMatrix,labelMatrix)
    plotBestFit(dataMatrix,labelMatrix,weights)

def classifyVector(inX,weights):
    prob = sigmoid(np.sum(inX*weights))
    if prob>0.5:return 1.0
    else: return 0.0

def colicTest(train_file,test_file):
    frTrain = open(train_file)
    frTest = open(test_file)
    trainSet = []
    trainLabel = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for n in currLine[:-1]:
            lineArr.append(float(n))
        trainSet.append(lineArr)
        trainLabel.append(float(currLine[-1]))
    trainSet = np.array(trainSet)
    trainLabel = np.array(trainLabel).reshape((trainSet.shape[0],1))
    trainWeights = stocGradAscent2(trainSet,trainLabel,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = [float(x) for x in line.strip().split('\t')]
        r = int(classifyVector(np.array(currLine[:-1]).reshape((1,-1)).T,trainWeights))
        if r != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of the test is:%f"%errorRate)
    return errorRate

def multiTest(train_file,test_file):
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorRate = colicTest(train_file,test_file)
        errorSum += errorRate
    print("after %d iterations the average error rate is:%f"%(numTests,errorSum/numTests))

if __name__ == "__main__":
    # ceshiLR()
    colicTest("horseColicTrain.txt","horseColicTest.txt")