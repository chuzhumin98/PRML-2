from MNN import *

trainSizeSet = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 60000]
evaluate = []
for i in range(1):
    usedTrainSize = trainSizeSet[i]
    sample = doMNN()
    evaluate.append(sample)

print(evaluate)

