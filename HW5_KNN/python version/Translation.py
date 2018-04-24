import KNN
import numpy as np
import math

def translation(imageSet):
    translateImageSet = []
    for i in range(len(imageSet)):
        columnweight = np.tile(list(range(KNN.column)),(KNN.row,1))
        rowweight = []
        for j in range(KNN.row):
            rowweight.append(np.tile(j, KNN.column))
        columntotal = np.sum(np.multiply(columnweight, imageSet[i]))
        columnavg = columntotal/np.sum(imageSet[i])
        rowtotal = np.sum(np.multiply(rowweight, imageSet[i]))
        rowavg = rowtotal/np.sum(imageSet[i])
        deltarow = (KNN.row-1)/2 - rowavg #delta row you should change
        deltacolumn = (KNN.column-1)/2 - columnavg #delta column you should change
        bottom = math.floor(deltarow)
        top = math.ceil(deltarow)
        right = math.ceil(deltacolumn)
        left = math.floor(deltacolumn)
        leftW = right - deltacolumn
        bottomW = top - deltarow
        wmatrix = [[bottomW*leftW, bottomW*(1-leftW)], [(1-bottomW)*leftW, (1-bottomW)*(1-leftW)]]
        #print(i,' row:',rowavg,',column:',columnavg)
        #print('bottom:',bottom,' left',left)
       #print(wmatrix)
        imagetranslate = np.tile(0,(KNN.row, KNN.column))
        for j in range(KNN.row):
            for k in range(KNN.column):
                if (j+bottom >=0 and j+bottom < KNN.row):
                    if (k+left >= 0 and k+left < KNN.column):
                        imagetranslate[j+bottom][k+left] += wmatrix[0][0]*imageSet[i][j][k]
                    if (k+right >= 0 and k+right < KNN.column):
                        imagetranslate[j+bottom][k+right] += wmatrix[0][1]*imageSet[i][j][k]
                if (j + top >= 0 and j + top < KNN.row):
                    if (k + left >= 0 and k + left < KNN.column):
                        imagetranslate[j + top][k + left] += wmatrix[1][0] * imageSet[i][j][k]
                    if (k + right >= 0 and k + right < KNN.column):
                        imagetranslate[j + top][k + right] += wmatrix[1][1] * imageSet[i][j][k]
        columntotal = np.sum(np.multiply(columnweight, imagetranslate))
        columnavg = columntotal / np.sum(imagetranslate)
        rowtotal = np.sum(np.multiply(rowweight, imagetranslate))
        rowavg = rowtotal / np.sum(imagetranslate)
        if ((i+1) % 100 == 0):
            print("has translate ",(i+1),' images, its rowavg:',rowavg,'columavg:',columnavg)
        translateImageSet.append(imagetranslate)
    return translateImageSet


def writeTranslateImage(imageSet, imagelabel, filePath):
    file = open(filePath, 'wb')
    np.save(file, imageSet)
    np.save(file, imagelabel)
    file.close()

# import data
[trainImage, trainLabel, testImage, testLabel] = KNN.importData()
translateTest = translation(testImage)
writeTranslateImage(translateTest, testLabel, 'output/test_translate.npy')
translateTrain = translation(trainImage)
writeTranslateImage(translateTrain, trainLabel, 'output/train_translate.npy')

