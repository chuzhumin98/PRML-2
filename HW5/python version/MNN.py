from ReadData import *

# 对每张图片进行所有训练集的p-范数距离求解
def distance(train, image, p):
    test = np.tile(image, (trainSize, 1, 1))
    delta = train - test
    if (p <= 150):
        delta = delta**p
        sums = np.sum(np.abs(delta),axis=(1,2))
        return sums**(1.0/p)
    else:
        return np.max(np.abs(delta), axis=(1,2))

[trainImage, trainLabel, testImage, testLabel] = importData()
distance(trainImage, testImage[0], 2)