import KNN
import numpy as np
import time
import random
import xlwt
import math


# MNN function, add for data load in
def doMNN(p,trainImage, trainLabel, testImage, testLabel, distanceMethod=0):
    # MNN
    time1 = time.time()
    count = 0
    for i in range(len(testImage)):
        if (distanceMethod == 0):
            dist = KNN.distance(trainImage, testImage[i], p)
        else:
            dist = tangentDistance(trainImage, testImage[i], p)
        nearest = np.argmin(dist)
        if (trainLabel[nearest] == testLabel[i]):
            count += 1
        print(i, ':', trainLabel[nearest], "-", testLabel[i],'with train size=',len(trainImage),'now ',count,'/',(i+1))
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
def tangentDistanceEachPair(image1, image2, p):
    rowvar1 = np.var(np.sum(image1, axis=0))
    rowvar2 = np.var(np.sum(image2, axis=0))
    columnvar1 = np.var(np.sum(image1, axis=1))
    columnvar2 = np.var(np.sum(image2, axis=1))
    # do row and column transfer
    if (rowvar1 > rowvar2):
        alpha = math.sqrt(rowvar1/rowvar2)
        image2_1 = rowTransfer(image2, alpha)
        image1_1 = image1
    else:
        alpha = math.sqrt(rowvar2/rowvar1)
        image1_1 = rowTransfer(image1, alpha)
        image2_1 = image2
    if (columnvar1 > columnvar2):
        alpha = math.sqrt(columnvar1/columnvar2)
        image2_2 = columnTransfer(image2_1, alpha)
        image1_2 = image1_1
    else:
        alpha = math.sqrt(columnvar2/columnvar1)
        image1_2 = columnTransfer(image1_1, alpha)
        image2_2 = image2_1
    delta = image2_2 - image1_2
    if (p <= 150):
        delta = delta**p
        sums = np.sum(np.abs(delta))
        return sums**(1.0/p)
    else:
        return np.max(np.abs(delta))



def tangentDistance(train, image, p):
    distarray = []
    for i in range(len(train)):
        distarray.append(tangentDistanceEachPair(train[i], image, p))
    #print(distarray)
    return distarray


[trainImage, trainLabel, testImage, testLabel] = sampleDataset(5000, 500)
doMNN(2, trainImage, trainLabel, testImage, testLabel, 1)

