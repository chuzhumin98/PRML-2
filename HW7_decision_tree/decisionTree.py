import scipy.io as scio
import treeNode
import numpy as np

#随机划分训练集、验证集和测试集
#    参数samples：所有样本数据
#    参数samplesLabels：样本所对应的标签
#    return：[训练集数据, 训练集标签, 验证集数据, 验证集标签, 测试集数据, 测试集标签]
def splitDatas(samples, samplesLabels):
    size = len(samplesLabels) #总的样本点个数
    indexArray = np.array(range(size), dtype=int) #下标数组
    np.random.shuffle(indexArray)
    validateStart = size * 3 // 5
    testStart = size * 4 // 5
    trainData = samples[indexArray[0:validateStart]]
    trainLabel = samplesLabels[indexArray[0:validateStart]]
    validateData = samples[indexArray[validateStart:testStart]]
    validateLabel = samplesLabels[indexArray[validateStart:testStart]]
    testData = samples[indexArray[testStart:size]]
    testLabel = samplesLabels[indexArray[testStart:size]]
    return [trainData, trainLabel, validateData, validateLabel, testData, testLabel]


#采用生成的决策树对结果进行预测
#    参数treeroot：决策树的树根节点
#    参数testData：测试数据
#    参数testLabel：测试数据的标签，用来计算准确率
#    return：[预测结果, 模型]
def Decision(treeroot, testData, testLabel):
    resultList = [] #记录预测结果的数组
    countRight = 0 #记录预测正确的数据点个数
    for i in range(len(testData)):
        tempNode = treeroot #记录现在滑动到的节点
        while (not tempNode.isleaf):
            if (testData[i][tempNode.column] > 0):
                tempNode = tempNode.trueNode
            else:
                tempNode = tempNode.falseNode
        resultList.append(tempNode.results)
        if (tempNode.results == testLabel[i][0]):
            countRight += 1
    return [resultList, countRight/len(testData)]


#对生长好的树进行剪枝操作
#    参数treeroot：生长出的树的树根
#    参数validateData：验证集上的数据
#    参数validateLabel：验证集上的标签
#    return：如果它满足上一层可以prune的条件，则返回分类正确的样本个数，否则返回-1
def Prune(treeroot, validateData, validateLabel):
    if (not treeroot.isleaf):
        #不是叶子节点时，先向下深探
        validateTrueData = validateData[validateData[:,treeroot.column] > 0]
        validateTrueLabel = validateLabel[validateData[:, treeroot.column] > 0]
        validateFalseData = validateData[validateData[:, treeroot.column] == 0]
        validateFalseLabel = validateLabel[validateData[:, treeroot.column] == 0]
        truePrune = Prune(treeroot.trueNode, validateTrueData, validateTrueLabel)
        falsePrune = Prune(treeroot.falseNode, validateFalseData, validateFalseLabel)
        if (truePrune >= 0 and falsePrune >= 0): #该节点满足
            thisNodeRight = len(validateLabel[validateLabel[:,0] == treeroot.results]) #剪枝后正确的个数
            if (truePrune + falsePrune < thisNodeRight): #剪枝后正确率提升则进行剪枝
                treeroot.trueNode = None
                treeroot.falseNode = None
                treeroot.isleaf = True
                print('prune to make correct num ',(truePrune+falsePrune),' -> ',thisNodeRight, ' of ',len(validateLabel))
                return thisNodeRight #上一层可能继续剪枝
            else:
                return -1 #上一层无法剪枝
        else:
            return -1
    else: #如果是叶子节点，则返回验证集上正确的个数，方便上一层判断能否剪枝
        thisNodeRight = len(validateLabel[validateLabel[:, 0] == treeroot.results])  #该节点上正确的个数
        return thisNodeRight


#导入数据
dataFile = 'Sogou_webpage.mat'
data = scio.loadmat(dataFile)
wordMat = data['wordMat']
doclabel = data['doclabel']
#划分训练集、验证集和测试集
trainData, trainLabel, validateData, validateLabel, testData, testLabel = splitDatas(wordMat, doclabel)
#进行决策树的生成
treeroot = treeNode.treeNode()
treeNode.GenerateTree(treeroot, trainData, trainLabel)
Prune(treeroot, validateData, validateLabel)

#测试准确率情况
results, accuracy = Decision(treeroot, trainData, trainLabel)
print('train set accuracy:',accuracy)
results, accuracy = Decision(treeroot, validateData, validateLabel)
print('validate set accuracy:',accuracy)
results, accuracy = Decision(treeroot, testData, testLabel)
print('test set accuracy:',accuracy)
print('test result:',results)

