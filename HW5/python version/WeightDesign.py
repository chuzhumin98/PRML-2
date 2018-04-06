from ReadData import *

# import data
[trainImage, trainLabel, testImage, testLabel] = importData()

# init the variable
imageGroupByLabel = []
for i in range(10):
    imageGroupByLabel.append([])

for i in range(len(trainImage)):
    imageGroupByLabel[trainLabel[i]].append(trainImage[i])

# each group's image number
imageLength = []
for i in range(10):
    imageLength.append(len(imageGroupByLabel[i])/len(trainImage))

# image mean of each group
imageMean = []
for i in range(10):
    imageMean.append(np.mean(imageGroupByLabel[i],axis=0))

# image variance of each group
imageVar = []
for i in range(10):
    imageVar.append(np.var(imageGroupByLabel[i],axis=0))

varBetweenClass = np.var(imageMean, axis=0)
print(varBetweenClass)