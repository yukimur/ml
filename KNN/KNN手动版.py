
import os
import numpy as np
import matplotlib.pyplot as plt
import operator

def data2matrix(filename):
    pass

def autoNorm(dataSet):
    """归一化特征值"""
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals+1

    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def autoNormSingle(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals+1
    normDataSet = dataSet/ranges
    return normDataSet

def classify0(testSet,testLabels,dataSet,labels,K):
    error = 0
    m = dataSet.shape[0]
    for inX,inY in zip(testSet,testLabels):
        diffMat = np.tile(inX,(m,1)) - dataSet
        sqDiffMat = diffMat**2
        distances = sqDiffMat.sum(axis=1)**0.5
        sortedDistances = distances.argsort()
        classCount = {}
        for i in range(K):
            voteIlabel = labels[sortedDistances[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        pre = sortedClassCount[0][0]
        if pre!=inY:
            error += 1
    return error,error / float(testSet.shape[0])

def readFile(fileName):
    """手写数字专用读函数"""
    dataSet = []
    labels = []
    files = os.listdir(fileName)
    for file in files:
        tmp = []
        labels.append(int(file.split('_')[0]))
        f = open(os.path.join(fileName,file))
        for line in f.readlines():
            tmp.extend([int(c) for c in line.strip()])
        dataSet.append(tmp)
    return np.array(dataSet),np.array(labels)

def ceshiSingle(inX,dataSet,labels,K):
    m = dataSet.shape[0]
    diffMat = np.tile(inX, (m, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    distances = sqDiffMat.sum(axis=1) ** 0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(K):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print("the predict Y is:",sortedClassCount[0][0])


def main(trainFile,testFile,K):
    trainSet_orig,trainLabels = readFile(trainFile)
    testSet_orig,testLabels = readFile(testFile)
    trainSet,trainRanges,trainMinVals = autoNorm(trainSet_orig)
    testSet,testRanges,testMinVals = autoNorm(testSet_orig)

    error,rate = classify0(testSet,testLabels,trainSet,trainLabels,K)
    print("the total error rate is: %f" % (rate))

    inX = input("请输入：")
    inX = np.array([int(x) for x in inX.strip()])
    inX = autoNormSingle(inX)
    ceshiSingle(inX,trainSet,trainLabels,K)

if __name__ == "__main__":
    trainFile = 'trainingDigits'
    testFile = 'testDigits'
    K = 3
    main(trainFile,testFile,K)