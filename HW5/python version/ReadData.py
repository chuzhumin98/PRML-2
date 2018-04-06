import numpy as np
import struct
import random

# set constant varibles
row = 28
column = 28
usedTestSize = 500

def importData(usedTrainSize = 10000):
    trainSize = 60000
    testSize = 10000
    #set files' path
    trainimagepath = "input/train_images"
    trainlabelpath = "input/train_labels"
    testimagepath = "input/test_images"
    testlabelpath = "input/test_labels"

    # state for data structure
    trainImage = []
    trainLabel = []
    testImage = []
    testLabel = []

    # read for train images
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

    # read for train labels
    f = open(trainlabelpath, 'rb')
    string = f.read()
    print('train label file size:', len(string))
    index = 8
    for i in range(trainSize):
        [label] = struct.unpack('B', string[index:index + 1])
        index += 1
        trainLabel.append(label)

   # print(trainLabel)

    # read for test images
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

    # read for test labels
    f = open(testlabelpath, 'rb')
    string = f.read()
    print('test label file size:', len(string))
    index = 8
    for i in range(testSize):
        [label] = struct.unpack('B', string[index:index + 1])
        index += 1
        testLabel.append(label)

    # shuffle for train set and test set
    train = []
    for i in range(trainSize):
        train.append([trainImage[i],trainLabel[i]])
    random.shuffle(train)

    test = []
    for i in range(testSize):
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

    #print(usedTrainLabel)

    return [np.array(usedTrainImage), np.array(usedTrainLabel), np.array(usedTestImage),
            np.array(usedTestLabel)]
