
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator


def file2matrix(filename):
    """数据转为矩阵"""
    fr = open(filename)
    line_list = fr.readlines()
    m = len(line_list)
    returnMat = np.zeros((m,3))
    classLabelVector = []
    for i,line in enumerate(line_list):
        listFromline  = line.strip().split('\t')
        returnMat[i,:] = listFromline[:3]
        classLabelVector.append(int(listFromline[-1]))
    return returnMat,classLabelVector

def autoNorm(dataSet):
    """归一化特征值"""
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals+1

    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def classify0(inX,dataSet,labels,k):
    m = dataSet.shape[0]
    diffMat = np.tile(inX,(m,1)) - dataSet
    sqDiffMat = diffMat**2
    distances = sqDiffMat.sum(axis=1)**0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def datingClassify():
    hoRatio = 0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)

    trainMat = normMat[numTestVecs:,:]
    trainLabel = datingLabels[numTestVecs:]
    errorCount = 0.0
    for i in range(numTestVecs):
        classify = classify0(normMat[i,:],trainMat,trainLabel,3)
        print("the classifier came back with:%d,the read anser is:%d"%(classify,datingLabels[i]))
        if (classify!=datingLabels[i]):
            errorCount += 1
    print("the total error rate is:%f"%(errorCount/float(numTestVecs)))

def clasdifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(input("percentage of time spent playing video games ?"))

if __name__ == "__main__":
    datingClassify()