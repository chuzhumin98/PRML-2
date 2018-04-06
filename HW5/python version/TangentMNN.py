import KNN
import numpy as np
import time
import random
import math

# MNN function, add for data load in
def doMNN(p,trainImage, trainLabel, testImage, testLabel):
    # MNN
    time1 = time.time()
    count = 0
    for i in range(len(testImage)):
        dist = KNN.distance(trainImage, testImage[i], p)
        nearest = np.argmin(dist)
        if (trainLabel[nearest] == testLabel[i]):
            count += 1
        if ((i+1)%10 == 0):
            print(i, ':', trainLabel[nearest], "-", testLabel[i],'with train size=',len(trainImage))
    time2 = time.time()
    print("has used time ", (time2 - time1), "s in MNN testing")
    print("has corrected ", count, "/", len(testImage), ",accuracy = ", count / len(testImage))
    return [(time2-time1), count/len(testImage)]

# load the data
filetest = open("output/test_translate.npy", "rb")
testImage = np.load(filetest)
testLabel = np.load(filetest)

filetrain = open("output/train_translate.npy", "rb")
trainImage = np.load(filetrain)
trainLabel = np.load(filetrain)

indexMax = max(math.floor(random.random()*len(testImage)), 500)
mytestImage = testImage[indexMax-500:indexMax]
mytestLabel = testLabel[indexMax-500:indexMax]
doMNN(2, trainImage, trainLabel, mytestImage, mytestLabel)
