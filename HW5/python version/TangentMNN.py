import KNN
import numpy as np


filetest = open("output/test_translate.npy", "rb")
testImage = np.load(filetest)
testLabel = np.load(filetest)

print(testImage[0])
print(testLabel)