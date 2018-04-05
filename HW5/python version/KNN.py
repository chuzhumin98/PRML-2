from ReadData import *
import time
import sys

# import data
[trainImage, trainLabel, testImage, testLabel] = importData()

# for every test image calculate its k-norm distance to train set
def distance(train, image, p):
    test = np.tile(image, (len(train), 1, 1))
    delta = train - test
    if (p <= 150):
        delta = delta**p
        sums = np.sum(np.abs(delta),axis=(1,2))
        return sums**(1.0/p)
    else:
        return np.max(np.abs(delta), axis=(1,2))

# MNN function
def doMNN(p,trainSize=10000):
    # import data
    #[trainImage, trainLabel, testImage, testLabel] = importData(trainSize)

    # MNN
    time1 = time.time()
    count = 0
    for i in range(usedTestSize):
        dist = distance(trainImage, testImage[i], p)
        nearest = np.argmin(dist)
        if (trainLabel[nearest] == testLabel[i]):
            count += 1
        if ((i+1)%10 == 0):
            print(i, ':', trainLabel[nearest], "-", testLabel[i],'with train size=',len(trainImage))
    time2 = time.time()
    print("has used time ", (time2 - time1), "s in MNN testing")
    print("has corrected ", count, "/", usedTestSize, ",accuracy = ", count / usedTestSize)
    return [(time2-time1), count/usedTestSize]


# kNN function [slove for one k value]
def doKNN(k):
    # import data
    #[trainImage, trainLabel, testImage, testLabel] = importData()

    # KNN
    time1 = time.time()
    count = 0
    for i in range(usedTestSize):
        dist = distance(trainImage, testImage[i], 2)
        voteNum = np.zeros(10) #the index i is for number i vote num
        for j in range(k): # for the top k nearest to vote
            nearest = np.argmin(dist)
            voteNum[trainLabel[nearest]] += 1
            dist[nearest] = sys.float_info.max #to avoid use one train point twice
        voteResult = np.argmax(voteNum)
        if (voteResult == testLabel[i]):
            count += 1
        if ((i+1)%10 == 0):
            print(i, ':', trainLabel[nearest], "-", testLabel[i],'with k=',k)
    time2 = time.time()
    print("has used time ", (time2 - time1), "s in MNN testing")
    print("has corrected ", count, "/", usedTestSize, ",accuracy = ", count / usedTestSize)
    return [(time2 - time1), count / usedTestSize]