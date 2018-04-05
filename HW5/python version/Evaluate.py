from MNN import *
import xlwt

trainSizeSet = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000]
evaluate = []
for i in range(len(trainSizeSet)):
    sample = doMNN(trainSizeSet[i])
    evaluate.append(sample)

print(evaluate)
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('MNN', cell_overwrite_ok=True)
for i in range(len(evaluate)):
    sheet.write(i, 0, trainSizeSet[i])
    sheet.write(i, 1, evaluate[i][0])
    sheet.write(i, 2, evaluate[i][1])

book.save("output/MNN_trainSize.xlsx")