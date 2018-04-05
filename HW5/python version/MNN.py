from ReadData import *
import time

# 对每张图片进行所有训练集的p-范数距离求解
def distance(train, image, p):
    test = np.tile(image, (usedTrainSize, 1, 1))
    delta = train - test
    if (p <= 150):
        delta = delta**p
        sums = np.sum(np.abs(delta),axis=(1,2))
        return sums**(1.0/p)
    else:
        return np.max(np.abs(delta), axis=(1,2))


# 导入数据
[trainImage, trainLabel, testImage, testLabel] = importData()

# MNN
time1 = time.time()
count = 0
for i in range(usedTestSize):
    dist = distance(trainImage, testImage[i], 2)
    nearest = np.argmin(dist)
    if (trainLabel[nearest] == testLabel[i]):
        count += 1
    print(i,':',trainLabel[nearest],"-",testLabel[i])
time2 = time.time()
print("has used time ",(time2-time1),"s in MNN testing")
print("has corrected ",count,"/",usedTestSize,",accuracy = ",count/usedTestSize)