from KNN import *
import time
import xlwt

[trainImage, trainLabel, testImage, testLabel] = importData()

def getScore():
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
        prob = len(imageGroupByLabel[i]) / len(trainImage)
        imageLength.append(prob)
        imageLengthMatrix.append(np.tile(prob, [row, column]))

    # image mean of each group
    imageMean = []
    for i in range(10):
        imageMean.append(np.mean(imageGroupByLabel[i], axis=0))

    # image variance of each group
    imageVar = []
    for i in range(10):
        imageVar.append(np.var(imageGroupByLabel[i], axis=0))

    varBetweenClass = np.var(imageMean, axis=0)

    vares = np.multiply(imageVar, imageLengthMatrix)
    varInClass = np.sum(vares, axis=0) + np.tile(0.00001, (row, column))

    # get each pixel's score
    score = np.divide(varBetweenClass, varInClass)
    return score


# for every test image calculate its weighted k-norm distance to train set
def weightDistance(train, image, p, score):
    test = np.tile(image, (len(train), 1, 1))
    scores = np.tile(score, (len(train), 1, 1))
    delta = train - test
    if (p <= 150):
        delta = delta**p
        weightPDist = np.multiply(np.abs(delta), scores)
        sums = np.sum(weightPDist,axis=(1,2))
        return sums**(1.0/p)
    else:
        return np.max(np.abs(delta), axis=(1,2))


# MNN function
def doWeightMNN(p, score, trainSize=10000):
    # import data
    [trainImage, trainLabel, testImage, testLabel] = importData(trainSize)

    # MNN
    time1 = time.time()
    count = 0
    for i in range(usedTestSize):
        dist = weightDistance(trainImage, testImage[i], p, score)
        nearest = np.argmin(dist)
        if (trainLabel[nearest] == testLabel[i]):
            count += 1
        if ((i+1)%10 == 0):
            print(i, ':', trainLabel[nearest], "-", testLabel[i],'with weighted train size=',len(trainImage))
    time2 = time.time()
    print("has used time ", (time2 - time1), "s in weighted MNN testing")
    print("has corrected ", count, "/", usedTestSize, ",accuracy = ", count / usedTestSize)
    return [(time2-time1), count/usedTestSize]


# done initMNN and weightMNN in one method ,to confirm their trainset is the same
def compareInitWeightModel(p, score, trainSize=10000):
    # import data
    [trainImage, trainLabel, testImage, testLabel] = importData(trainSize)

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
    evaluate = [(time2 - time1), count / usedTestSize]

    # weighted MNN
    time1 = time.time()
    count = 0
    for i in range(usedTestSize):
        dist = weightDistance(trainImage, testImage[i], p, score)
        nearest = np.argmin(dist)
        if (trainLabel[nearest] == testLabel[i]):
            count += 1
        if ((i+1)%10 == 0):
            print(i, ':', trainLabel[nearest], "-", testLabel[i],'with weighted train size=',len(trainImage))
    time2 = time.time()
    print("has used time ", (time2 - time1), "s in weighted MNN testing")
    print("has corrected ", count, "/", usedTestSize, ",accuracy = ", count / usedTestSize)
    evaluate.append((time2-time1))
    evaluate.append(count/usedTestSize)
    return evaluate

def evaluaeWeightModel(score):
    trainSizeSet = [5000, 10000, 20000, 60000]
    evaluate = []
    for i in range(len(trainSizeSet)):
        sample = compareInitWeightModel(2, score, trainSizeSet[i])
        print(sample)
        evaluate.append(sample)
    print(evaluate)
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('MNN', cell_overwrite_ok=True)
    for i in range(len(evaluate)):
        sheet.write(i, 0, trainSizeSet[i])
        sheet.write(i, 1, evaluate[i][0])
        sheet.write(i, 2, evaluate[i][1])
        sheet.write(i, 3, evaluate[i][2])
        sheet.write(i, 4, evaluate[i][3])

    book.save("output/MNN_weightModel.xls")

score = getScore()
evaluaeWeightModel(score)


