from KNN import *
import xlwt

# evaluate MNN along different train size
def evaluaeTrainSize():
    trainSizeSet = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000]
    evaluate = []
    for i in range(len(trainSizeSet)):
        sample = doMNN(2,trainSizeSet[i])
        evaluate.append(sample)

    print(evaluate)
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('MNN', cell_overwrite_ok=True)
    for i in range(len(evaluate)):
        sheet.write(i, 0, trainSizeSet[i])
        sheet.write(i, 1, evaluate[i][0])
        sheet.write(i, 2, evaluate[i][1])

    book.save("output/MNN_trainSize.xls")


# evaluate KNN with different k value
def evaluateKValue():
    kSet = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    evaluate = []
    for i in range(len(kSet)):
        if (i != 0):
            sample = doKNN(2, kSet[i])
        else:
            sample = doMNN() #when k=1, use method doMNN()
        evaluate.append(sample)

    print(evaluate)
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('KValue', cell_overwrite_ok=True)
    for i in range(len(evaluate)):
        sheet.write(i, 0, kSet[i])
        sheet.write(i, 1, evaluate[i][0])
        sheet.write(i, 2, evaluate[i][1])

    book.save("output/KNN_KValue.xls")


# evulate MNN along different distance standard
def evaluaeDistanceNormP():
    normPSet = [1, 2, 3, 4, 5, 10, 20, 200]
    evaluate = []
    for i in range(len(normPSet)):
        sample = doMNN(normPSet[i])
        evaluate.append(sample)

    print(evaluate)
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('PNorm', cell_overwrite_ok=True)
    for i in range(len(evaluate)):
        sheet.write(i, 0, normPSet[i])
        sheet.write(i, 1, evaluate[i][0])
        sheet.write(i, 2, evaluate[i][1])

    book.save("output/MNN_NormP.xls")


evaluaeDistanceNormP()