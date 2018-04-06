import KNN
import numpy as np
import time
import random
import xlwt
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


# sample for translational data
def sampleDataset(usedTrainSize=10000, usedTestSize=500):
    # load the data
    filetest = open("output/test_translate.npy", "rb")
    testImage = np.load(filetest)
    testLabel = np.load(filetest)

    filetrain = open("output/train_translate.npy", "rb")
    trainImage = np.load(filetrain)
    trainLabel = np.load(filetrain)

    # shuffle for train set and test set
    train = []
    for i in range(len(trainImage)):
        train.append([trainImage[i], trainLabel[i]])
    random.shuffle(train)

    test = []
    for i in range(len(testImage)):
        test.append([testImage[i], testLabel[i]])
    random.shuffle(test)

    # set for KNN request variable
    usedTrainImage = []
    usedTestImage = []
    usedTrainLabel = []
    usedTestLabel = []

    # set the value for varibles
    for i in range(usedTrainSize):
        usedTrainImage.append(train[i][0])
        usedTrainLabel.append(train[i][1])

    for i in range(usedTestSize):
        usedTestImage.append(test[i][0])
        usedTestLabel.append(test[i][1])

    # print(usedTrainLabel)

    return [np.array(usedTrainImage), np.array(usedTrainLabel), np.array(usedTestImage),
            np.array(usedTestLabel)]


# evaluate MNN along different train size
def evaluaeTranslateTrainSize():
    trainSizeSet = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000]
    evaluate = []
    for i in range(len(trainSizeSet)):
        [trainImage, trainLabel, testImage, testLabel] = sampleDataset(usedTrainSize=trainSizeSet[i])
        sample = doMNN(2, trainImage, trainLabel, testImage, testLabel)
        evaluate.append(sample)

    print(evaluate)
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('MNN', cell_overwrite_ok=True)
    for i in range(len(evaluate)):
        sheet.write(i, 0, trainSizeSet[i])
        sheet.write(i, 1, evaluate[i][0])
        sheet.write(i, 2, evaluate[i][1])

    book.save("output/MNN_translate_trainSize.xlsx")

# transfer row length with parameter alpha
def rowTransfer(image, alpha):
    transformImage = np.tile(0, (KNN.row, KNN.column))
    base = 13.5 #the center base point
    for i in range(KNN.row):
        middle = (i - base)*alpha + base
        low = math.floor(middle)
        high = math.ceil(middle)
        plow = high-middle
        if (low >= 0 and low < KNN.row):
            transformImage[low] = np.add(transformImage[low], np.multiply(image[i], plow))
        if (high >= 0 and high < KNN.row):
            transformImage[high] = np.add(transformImage[high], np.multiply(image[i], 1-plow))
    return transformImage


# transfer column length with parameter alpha
def columnTransfer(images, alpha):
    transformImage = np.tile(0, (KNN.row, KNN.column))
    image = np.transpose(images)
    base = 13.5 #the center base point
    for i in range(KNN.column):
        middle = (i - base)*alpha + base
        low = math.floor(middle)
        high = math.ceil(middle)
        plow = high-middle
        if (low >= 0 and low < KNN.column):
            transformImage[low] = np.add(transformImage[low], np.multiply(image[i], plow))
        if (high >= 0 and high < KNN.column):
            transformImage[high] = np.add(transformImage[high], np.multiply(image[i], 1-plow))
    return np.transpose(transformImage)


# calcaulate two images' tangent distance
def tangentDistance(image1, image2, p):
    rowsum1 = np.sum(image1, axis=0)
    rowsum2 = np.sum(image1, axis=0)



[trainImage, trainLabel, testImage, testLabel] = sampleDataset()
print(columnTransfer(trainImage[0], 3.0))

