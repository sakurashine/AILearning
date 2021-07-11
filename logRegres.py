#coding=utf-8
import os, sys  #用于读取本地数据集文件时获取当前路径
from numpy import *
import matplotlib.pyplot as plt


#加载数据集，数据集的前两个值分别为X1和X2,第三个值是数据对应的类别标签，
def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open(sys.path[0]+'/testSet.txt')
	for line in fr.readlines():  #逐行读取
		lineArr = line.strip().split()
		#因为线性回归化式为 H(x) = W0 + W1*X1 + W2*X2即为 (W0, W1, W2)*(1, X1, X2)，为了方便计算，把X0的值设置成了1.0
		#其中 (W0, W1, W2) 即为所求回归系数 W。 为了方便计算, 读出 X1, X2 后要在前面补上一个 1.0
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  #append() 方法用于在列表末尾添加新的对象
		labelMat.append(int(lineArr[2]))  
	return dataMat,labelMat


#分类器的分类（转换）函数
def sigmoid(inX):
	return 1.0/(1+exp(-inX))  #计算 sigmoid 函数


#梯度上升算法，用来计算出最佳回归系数
def gradAscent(dataMatIn, classLabels):
	'''第一个参数是2维数组，每列代表每个不同特征，每行代表每个训练样本
       第二个参数是类别标签，1*100的行向量，为便于计算，将其转换为列向量，即进行转置，并赋值给labelMat
    '''
	dataMatrix = mat(dataMatIn)  #获得输入数据并将其转换为Numpy矩阵数据类型
	labelMat = mat(classLabels).transpose()  #转换为NumPy矩阵数据类型
	m,n = shape(dataMatrix)  #shape函数是numpy.core.fromnumeric中的函数，它的功能是查看矩阵或者数组的维数
	alpha = 0.001  #步长，向函数增长最快的方向的移动量，即学习率
	maxCycles = 500  #迭代次数
	weights = ones((n,1))  #生成n行一列的元素为1的矩阵赋给weights，即回归系数初始化为1

	#循环 maxCycles次, 每次都沿梯度向真实值 labelMat 靠拢
	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)  #选取一个样本
		error = (labelMat - h)  #计算误差
		weights = weights + alpha * dataMatrix.transpose() * error  #更新回归系数
	return weights


'''随机梯度上升算法，相比较于梯度上升算法占用更少的计算资源，h和error由向量格式变成了数值格式，
   且没有了矩阵的转换过程，所有变量的数据类型都是NumPy数组，但是因为此算法只选取了一个数据，
   迭代了一次，比不上迭代了500次的梯度上升算法那般稳定的收敛性，所以拟合效果并不完美。
   随机梯度上升是一个在线算法，它可以在新数据到来时就完成参数更新，而不需要重新读取整个数据集来进行批处理运算
'''
def stocGradAscent0(dataMatrix, classLabels):
	m, n = shape(dataMatrix)
	alpha = 0.01
	weights = ones(n)
	for i in range(m):
		h = sigmoid(sum(dataMatrix[i]*weights))
		error = classLabels[i] - h 
		weights = weights + alpha * error * dataMatrix[i]
	return weights


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


#输出运用几种算法后得到的最理想的回归系数的值并画图
def GetResult():
    dataArr, labelMat = loadDataSet()

    #运用梯度上升算法
    weights1 = gradAscent(dataArr, labelMat).getA()#使用getA()方法将矩阵类型的weights转换为数组类型
    #运用随机梯度上升算法
    weights2 = stocGradAscent0(array(dataArr), labelMat)
    #运用改进的随机梯度上升算法
    weights3 = stocGradAscent1(array(dataArr), labelMat)

    plotBestFit(weights1,weights2,weights3)  #画图

    
#画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights1,weights2,weights3):
	#weights = wei.get()
	dataMat,labelMat = loadDataSet()
	dataArr = array(dataMat)
	n = shape(dataArr)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if int(labelMat[i]) == 1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	
	#绘制散点图
	fig = plt.figure()
	ax = fig.add_subplot(111)  #1x1网格，第一子图
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  
	ax.scatter(xcord2, ycord2, s=30, c='green')  
	#画线
	x = arange(-3.0, 3.0, 0.1) 
	#y = (-('weights'+'i')[0]-('weights'+'i')[1]*x)/('weights'+'i')[2]
	y1 = (-weights1[0]-weights1[1]*x)/weights1[2]
	y2 = (-weights2[0]-weights2[1]*x)/weights2[2]
	y3 = (-weights3[0]-weights3[1]*x)/weights3[2]
	ax.plot(x, y1, color = 'k', label = 'gradAscent')  #plot()函数画二维线
	ax.plot(x, y2, color = 'b', label = 'stocGradAscent0')
	ax.plot(x, y3, color = 'm', label = 'stocGradAscent1')
	plt.xlabel('X1')
	plt.ylabel('X2');
	plt.legend()  #为图表添加标注
	plt.show()  #显示图像


#主函数
if __name__ == '__main__':
    GetResult()