# coding:utf-8

#用决策树进行分类判定鱼类和非鱼类的算法
import operator  #包含了一些操作符函数，用于majorityCnt()方法中作排序等运算
from math import log  #log方法是math库里面的，调出来算对数


#海洋动物数据集
def createDataSet():
	#每条实例三个值，分别对应：不浮出水面可以生存？是1否0；有脚蹼？是1否0；属于鱼类？是yes否no
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    #两个类标签：不浮出水面是否可以生存；是否有脚蹼
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


#计算香农熵
def calcShannonEnt(dataSet):
	#数据集中实例总数，此处为5
    numEntries = len(dataSet)
    #定义一个类别统计dict字典对象，统计所有类别出现的次数，用于计算所有类别出现的概率，进而计算香农熵
    labelCounts = {}
    
    #以行遍历数据集
    for featVec in dataSet:
        # 储存当前实例的类标签（每行数据的最后一个数据是类标签）
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        #print featVec, labelCounts

    #初始化香农熵
    shannonEnt = 0.0
    #遍历labelCounts对象，labelCounts[i]代表第i+1个类标签的出现次数
    for key in labelCounts:
        #prob即probability，类标签出现概率
        prob = float(labelCounts[key])/numEntries
        # 计算香农熵，对类标签概率以2为底求对数
        shannonEnt -= prob * log(prob, 2)#log(x,base)：x是数值表达式，base是底数，默认为自然对数e
    return shannonEnt


#以指定特征按数据进行划分（目的是为了之后对所有特征遍历，求出最高信息增益的特征加以选择）
def splitDataSet(dataSet, index, value):
    """splitDataSet通过遍历dataSet数据集，返回指定特征index所在列的值为value的行，去掉了index所在的那一列
        即依据index列进行分类，如果index列的数据等于value的时候，就要将index划分到我们创建的新的数据集中
        dataSet 待划分的数据集
        index 划分数据集的特征
        value 指定特征的值
    """

    #为了不对原始数据集做修改，创建一个新的数据集列表对象
    retDataSet = []
    #对数据集按行遍历
    for featVec in dataSet: 
        # 判断index列的值是否为value
        if featVec[index] == value:
            # featVec[:index]表示取featVec的前index行
            reducedFeatVec = featVec[:index]
            #list.append(obj)将obj看做一个对象添加进列表后
            #list.extend(obj)将obj看做一个序列合并至列表后
            # [index+1:]表示从第index+1行开始截取后面的数据
            reducedFeatVec.extend(featVec[index+1:]) 
            # 收集满足index列值为value的那些行，该行排除了index列
            #为什么排除index列，是因为按我们的目的，是要按照特征来划分数据。特征是划分手段，数据是划分结果
            retDataSet.append(reducedFeatVec)
    return retDataSet


#遍历整个数据集，循环计算香农熵和splitDataSet()函数，找到最好的划分方式
def chooseBestFeatureToSplit(dataSet):
    # 求第一行有多少列的特征, 减1是因为最后一列是label类标签
    numFeatures = len(dataSet[0]) - 1
    #保存原始香农熵，用于之后计算划分前后的信息增益（数据无序度的变化程度），挑最大的那个特征来划分
    baseEntropy = calcShannonEnt(dataSet)
    #初始化最优的信息增益值、信息增益率和最优的特征索引
    bestInfoGain, bestInfoGainRate, bestFeature = 0.0, 0.0, -1
    #遍历数据集中的所有特征
    for i in range(numFeatures):
        ''' 对featList = [example[i] for example in dataSet]这一句的理解：
            将dataSet中的数据按行依次放入example中，然后取得example中的example[i]元素放入featList中
            即，获取每个实例的第i+1个feature，组成list集合
            比如i=0时，返回[1, 1, 1, 0, 0]，也就是第一个特征列的所有值；i=-1时，返回所有类标签
        '''
        featList = [example[i] for example in dataSet]
        #set和list很像，区别是set中的数据每个值互不相同，这句代码是使用set对list数据进行去重
        uniqueVals = set(featList)
        #初始化一个临时的信息熵
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值，对每个特征划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算划分前后的信息增益（数据无序度的变化程度），挑信息增益最大的那个特征来划分
        infoGain = baseEntropy - newEntropy
 

        #ID3算法，使用信息增益来选择特征
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        # 若按某特征划分后，若infoGain大于bestInfoGain，则infoGain对应的特征分类区分样本的能力更强，更具有代表性
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            #将最大的信息增益对应的特征下标赋给bestFeature
            bestFeature = i

        '''#C4.5算法
        #与ID3算法的主要区别：不使用信息增益，而是使用信息增益率来选择特征
        #定义分裂信息度量
        splitInfo = calcShannonEnt(subDataSet)
        #若特征值相同，则跳过该特征
        if splitInfo == 0:
            continue
        #计算信息增益率
        InfoGainRate = infoGain/splitInfo
        if(InfoGainRate>bestInfoGainRate):
            bestInfoGainRate = InfoGainRate
            bestFeature = i
        '''

    #返回最优特征的索引
    return bestFeature


#选择出现次数最多的分类。classList参数：类标签集合
def majorityCnt(classList):
    #初始化一个统计类标签出现频率的dict字典对象
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #print sortedClassCount
    #返回出现次数最多的分类名称
    return sortedClassCount[0][0]


#创建树的数据结构
def createTree(dataSet, labels):
    #所有类标签的集合（代码解释见86行）
    classList = [example[-1] for example in dataSet]
    #第一个停止条件：如果数据集的最后一列的第一个值出现的次数=整个集合的长度，也就说只有一个类别，那么直接返回该类标签
    #count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #如果数据集只有1列，那么取最初出现label次数最多的一类，作为结果
    #第二个停止条件: 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    #选择最优特征的索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #获取最优特征的名称
    bestFeatLabel = labels[bestFeat]
    #初始化树
    myTree = {bestFeatLabel: {}}
    #labels列表是可变对象，在python函数中作为参数时传址引用，能被全局修改
    del(labels[bestFeat]) 
    #取出最优列，然后它的分支做分类
    featValues = [example[bestFeat] for example in dataSet]
    #对最优列去重
    uniqueVals = set(featValues)
    for value in uniqueVals:
        #求出剩余的标签label，labels[:]意为创建labels的一份副本
        subLabels = labels[:]
        #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


#使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    """ classify()对给定输入的结点进行分类
        inputTree  传入的决策树模型
        featLabels 特征名称
        testVec    测试输入的数据
        classLabel 返回结果，类标签名称
    """
    # 获取树的根节点key值,keys()[0]返回字典的第一个键值，此处为no surfacing
    firstStr = inputTree.keys()[0]
    #print(firstStr)
    
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 找到当前列表中第一个匹配firstStr变量的元素位置
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


#判定鱼类和非鱼类
def fishTest():
    #数据集，类标签集
    myDat, labels = createDataSet()
    print myDat
    print labels

    #计算类标签的香农熵
    calcShannonEnt(myDat)

    #计算最好的信息增益的列
    print chooseBestFeatureToSplit(myDat)

    import copy
    #labels列表是可变对象，在python函数中作为参数时传址引用，能被全局修改。为避免这种情况，使用深拷贝
    #浅拷贝只拷贝父对象，深拷贝完全拷贝了父对象及其子对象
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    #[1, 1]表示要取的分支上的节点位置
    print(classify(myTree, labels, [1, 1]))
    

#主函数
if __name__ == '__main__':
    fishTest()