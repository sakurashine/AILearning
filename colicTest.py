#coding=utf-8
import os, sys
from numpy import *
import numpy as np

#分类器的分类（转换）函数
def sigmoid(inX):
    if inX >= 0:  #计算 sigmoid 函数，且避免极大数据溢出
        return 1.0/(1+exp(-inX))
    else:
        return exp(inX)/(1+exp(inX)) 


#改进的随机梯度上升算法，随机选取样本更新回归系数，且每次迭代都调整alpha
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


#以回归系数和特征向量作为输入来计算对应的sigmoid值，大于0.5返回1，小于0.5返回0
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0
 
 
#打开测试集和训练集，并对数据进行格式化处理，以及对缺失项的处理
def colicTest():
    frTrain=open(sys.path[0]+'/horseColicTraining.txt')
    frTest=open(sys.path[0]+'/horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(np.array(trainingSet),trainingLabels,500)
    errorCount=0
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=(float(errorCount)/numTestVec)
    print("the error rate of this test is:%f" % errorRate)
    return errorRate
 
 
#调用colicTest()10次并求结果的平均值
def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests)))


#主函数
if __name__ == '__main__':
    multiTest()