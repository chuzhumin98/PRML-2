import numpy as np
import struct
import random

# 设置一些常量的值
row = 28
column = 28
usedTrainSize = 6000
usedTestSize = 1000

def importData():
    trainSize = 60000
    testSize = 10000
    # 给出各文件的路径
    trainimagepath = "input/train_images"
    trainlabelpath = "input/train_labels"
    testimagepath = "input/test_images"
    testlabelpath = "input/test_labels"

    # 申明一些数据结构
    trainImage = []
    trainLabel = []
    testImage = []
    testLabel = []

    # 读入训练样本
    f = open(trainimagepath, 'rb')
    string = f.read()
    print('train image file size:', len(string))
    index = 16
    for i in range(trainSize):
        oneimage = []
        for j in range(row):
            onerow = np.zeros(row)
            [onerow[0], onerow[1], onerow[2], onerow[3], onerow[4], onerow[5], onerow[6],
             onerow[7], onerow[8], onerow[9], onerow[10], onerow[11], onerow[12], onerow[13],
             onerow[14], onerow[15], onerow[16], onerow[17], onerow[18], onerow[19], onerow[20],
             onerow[21], onerow[22], onerow[23], onerow[24], onerow[25], onerow[26], onerow[27]] = struct.unpack('28B',
                                                                                                                 string[
                                                                                                                 index:index + 28])
            index += 28
            oneimage.append(onerow)
        imagematrix = np.mat(oneimage)
        trainImage.append(imagematrix)
        if ((i + 1) % 10000 == 0):
            print("has read ", (i + 1), " images in train file")

    # 读入训练标签
    f = open(trainlabelpath, 'rb')
    string = f.read()
    print('train label file size:', len(string))
    index = 8
    for i in range(trainSize):
        [label] = struct.unpack('B', string[index:index + 1])
        index += 1
        trainLabel.append(label)

   # print(trainLabel)

    # 读入测试样本
    f = open(testimagepath, 'rb')
    string = f.read()
    print('test image file size:', len(string))
    index = 16
    for i in range(testSize):
        oneimage = []
        for j in range(row):
            onerow = np.zeros(row)
            [onerow[0], onerow[1], onerow[2], onerow[3], onerow[4], onerow[5], onerow[6],
             onerow[7], onerow[8], onerow[9], onerow[10], onerow[11], onerow[12], onerow[13],
             onerow[14], onerow[15], onerow[16], onerow[17], onerow[18], onerow[19], onerow[20],
             onerow[21], onerow[22], onerow[23], onerow[24], onerow[25], onerow[26], onerow[27]] = struct.unpack('28B',
                                                                                                                 string[
                                                                                                                 index:index + 28])
            index += 28
            oneimage.append(onerow)
        imagematrix = np.mat(oneimage)
        testImage.append(imagematrix)
        if ((i + 1) % 10000 == 0):
            print("has read ", (i + 1), " images in test file")

    # 读入测试标签
    f = open(testlabelpath, 'rb')
    string = f.read()
    print('test label file size:', len(string))
    index = 8
    for i in range(testSize):
        [label] = struct.unpack('B', string[index:index + 1])
        index += 1
        testLabel.append(label)

    # 对训练集和测试集进行打乱
    train = []
    for i in range(trainSize):
        train.append([trainImage[i],trainLabel[i]])
    random.shuffle(train)

    test = []
    for i in range(testSize):
        test.append([testImage[i], testLabel[i]])
    random.shuffle(test)

    # 需要在kNN中使用的量
    usedTrainImage = []
    usedTestImage = []
    usedTrainLabel = []
    usedTestLabel = []

    # 将这些量进行赋值
    for i in range(usedTrainSize):
        usedTrainImage.append(train[i][0])
        usedTrainLabel.append(train[i][1])

    for i in range(usedTestSize):
        usedTestImage.append(test[i][0])
        usedTestLabel.append(test[i][1])

    print(usedTrainLabel)

    return [np.array(usedTrainImage), np.array(usedTrainLabel), np.array(usedTestImage), np.array(usedTestLabel)]
