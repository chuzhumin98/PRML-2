from ReadData import *

# import data
[trainImage, trainLabel, testImage, testLabel] = importData()

# init the variable
imageGroupByLabel = []
for i in range(10):
    imageGroupByLabel.append([])

for i in range(len(trainImage)):
    imageGroupByLabel[trainLabel[i]].append(trainImage[i])

# each group's image number/totalLen
imageLength = []
imageLengthMatrix = []
for i in range(10):
    prob = len(imageGroupByLabel[i])/len(trainImage)
    imageLength.append(prob)
    imageLengthMatrix.append(np.tile(prob, [row, column]))

# image mean of each group
imageMean = []
for i in range(10):
    imageMean.append(np.mean(imageGroupByLabel[i],axis=0))

# image variance of each group
imageVar = []
for i in range(10):
    imageVar.append(np.var(imageGroupByLabel[i],axis=0))

varBetweenClass = np.var(imageMean, axis=0)

vares = np.multiply(imageVar, imageLengthMatrix)
varInClass = np.sum(vares, axis=0)+np.tile(0.00001, (row, column))

score = np.divide(varBetweenClass, varInClass)
print(score)