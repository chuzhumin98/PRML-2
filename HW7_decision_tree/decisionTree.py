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
