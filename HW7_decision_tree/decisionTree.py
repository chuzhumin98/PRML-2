import scipy.io as scio
import treeNode

dataFile = 'Sogou_webpage.mat'
data = scio.loadmat(dataFile)
wordMat = data['wordMat']
print(len(wordMat))
print(len(wordMat[0]))
print(wordMat[0][51:100])

doclabel = data['doclabel']
print(len(doclabel))
#print(doclabel)
treeroot = treeNode.treeNode()
treeNode.GenerateTree(treeroot, wordMat, doclabel)
