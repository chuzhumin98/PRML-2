import numpy as np
import math

labeltypes = 9 #总的样本类型的个数

totalLeafCount = 0 #记录分配到叶子的总节点数

# 决策树中的节点
class treeNode:
    def __init__(self, column = -1, results = None, isleaf = True, trueNode = None, falseNode = None):
        self.column = column #该节点分类所使用的特征index
        self.results = results #该节点分类的结果(每个节点都存一下，方便后期进行剪枝）
        self.isleaf = isleaf #记录该节点是否为叶子节点
        self.trueNode = trueNode #true分支，即该term在文中出现
        self.falseNode = falseNode #false分支，即该term在文中未出现


# 针对概率计算部分熵
def getEntropy(prob):
    if (prob < 1e-10): #概率太小时直接取0
        return 0
    else:
        return -prob*math.log2(prob)


# 根据所给的labels分布得到该类别下的分类结果
def getLabel(samplesLabels):
    labels = np.array(samplesLabels)
    labels = np.reshape(labels, (-1))
    labelsCount = []  # 记录各个标签样本点的个数
    for i in range(1, labeltypes + 1):
        countI = len(labels[labels == i])
        labelsCount.append([countI])
    return np.argmax(labelsCount)+1


# 不纯度度量
#    参数samplesLabels: 计算不纯度的样本标签
#    参数method：所使用的计算不纯度的方法，1为熵度量，2为错分度量，其他为Gini系数（default）
#    return: 不纯度度量的float值
def Impurity(samplesLabels, method=0):
    if (len(samplesLabels) == 0):
        return 0 #当不存在样本点时，不纯度则为0
    labels = np.array(samplesLabels)
    labels = np.reshape(labels, (-1))
    labelsCount = [] #记录各个标签样本点的个数
    for i in range(1,labeltypes+1):
        countI = len(labels[labels == i])
        labelsCount.append([countI])
    labelsCount = np.divide(labelsCount, len(labels)) #转换为概率
    #print(labelsCount)
    if (method == 1):
        # 进行熵度量的计算
        entropys = np.apply_along_axis(getEntropy, 1, labelsCount)
        totalEntropy = np.sum(entropys)
        return totalEntropy
    elif (method == 2):
        return 1 - np.max(labelsCount)
    else:
        return 1 - np.sum(np.square(labelsCount))


#对当前树上某个节点，选择待分特征
#    参数samplesUnderThisNode：该节点上的样本们
#    参数samplesLabels：这些样本所对应的标签
#    参数therthod：停止分支的信息增益阈值
#    参数therthodImpure：初始不纯度的停止分支阈值
#    参数method：所使用的计算不纯度的方法，1为熵度量，2为错分度量，其他为Gini系数（default）
#    return：分类所采用的特征,-1表示停止分类
def SelectFeature(samplesUnderThisNode, samplesLabels, thershod, thershodImpure, method=0):
    predScore = Impurity(samplesLabels, method) #初始的不纯度
    if (predScore < thershodImpure):
        global totalLeafCount
        totalLeafCount += len(samplesLabels)
        print('arrived leaf node with sample num ',len(samplesLabels),' with leaf count ',totalLeafCount)
        return -1
    gains = [] #信息增益程度
    for i in range(len(samplesUnderThisNode[0])): #对每一维度特征进行信息增益计算
        labelTrue = samplesLabels[samplesUnderThisNode[:,i] > 0] #拥有该词汇的labels
        labelFalse = samplesLabels[samplesUnderThisNode[:,i] == 0] #不拥有该词汇的labels
        succScore = (len(labelFalse)*Impurity(labelFalse,method)+len(labelTrue)*Impurity(labelTrue,method))/len(samplesLabels)
        gains.append(predScore-succScore)
    argmax = np.argmax(gains)
    max = gains[argmax]
    if (max < thershod):
        global totalLeafCount
        totalLeafCount += len(samplesLabels)
        print('arrived leaf node with sample num ',len(samplesLabels),' with leaf count ',totalLeafCount)
        return -1
    else:
        print('gain max ', max, ' in feature ', argmax, ' with sample num ', len(samplesLabels))
        return argmax

#对当前树上某个节点，进行节点切分
#    参数samplesUnderThisNode：该节点上的样本们
#    参数samplesLabels：这些样本所对应的标签
#    参数therthod：停止分支的信息增益阈值
#    参数therthodImpure：初始不纯度的停止分支阈值
#    参数method：所使用的计算不纯度的方法，1为熵度量，2为错分度量，其他为Gini系数（default）
#    return：如果已达阈值，则返回None；否则返回一个元组[feature column, true类的samples, false类的samples, true类的labels, false类的labels]
def SplitNode(samplesUnderThisNode, samplesLabels, thershod, thershodImpure, method=0):
    splitColumn = SelectFeature(samplesUnderThisNode, samplesLabels, thershod, thershodImpure, method)
    if (splitColumn == -1):
        return None
    else:
        samplesTrue = samplesUnderThisNode[samplesUnderThisNode[:,splitColumn] > 0]
        samplesFalse = samplesUnderThisNode[samplesUnderThisNode[:,splitColumn] == 0]
        labelsTrue = samplesLabels[samplesUnderThisNode[:,splitColumn] > 0]
        labelsFalse = samplesLabels[samplesUnderThisNode[:,splitColumn] == 0]
        return [splitColumn, samplesTrue, samplesFalse, labelsTrue, labelsFalse]


#决策树的生成算法
#    参数treeRoot：树根节点
#    参数samplesUnderThisNode：该节点上的样本们
#    参数samplesLabels：这些样本所对应的标签
#    参数therthod：停止分支的信息增益阈值
#    参数therthodImpure：初始不纯度的停止分支阈值
#    参数method：所使用的计算不纯度的方法，1为熵度量，2为错分度量，其他为Gini系数（default）
def GenerateTree(treeRoot, samples, samplesLabels, thershod=0.05, therthodImpure=0.2, method=0):
    splitResults = SplitNode(samples, samplesLabels, thershod, therthodImpure, method)
    treeRoot.results = getLabel(samplesLabels) #获得该节点的分类结果
    if (splitResults == None): #到达叶子节点即该分支生成完毕
        treeRoot.isleaf = True
        return
    else:
        #设置该节点的相关属性值
        treeRoot.column = splitResults[0]
        treeRoot.isleaf = False
        #生成两类的子节点
        trueNode = treeNode() #含有该term的节点
        falseNode = treeNode() #不含该term的节点
        #在该节点和子节点间建立连接
        treeRoot.trueNode = trueNode
        treeRoot.falseNode = falseNode
        #继续扩展两个子节点，尽可能进行扩展
        GenerateTree(trueNode, splitResults[1], splitResults[3], thershod, therthodImpure, method)
        GenerateTree(falseNode, splitResults[2], splitResults[4], thershod, therthodImpure, method)
