import numpy as np
import struct

# 给出各文件的路径
trainimagepath = "input/train_images"
trainlabelpath = "input/train_labels"
testimagepath = "input/test_images"
testlabelpath = "input/test_labels"

# 设置一些常量的值
trainSize = 60000
testSize = 10000
row = 28
column = 28

#申明一些数据结构
trainImage = []
trainLabel = []
testImage = []
testLabel = []

# 读入训练样本
f = open(trainimagepath, 'rb')
string = f.read()
print('train image file size:',len(string))
index=16
for i in range(trainSize):
    oneimage = []
    for j in range(row):
        onerow = np.zeros(row)
        [onerow[0], onerow[1], onerow[2], onerow[3], onerow[4], onerow[5], onerow[6],
         onerow[7], onerow[8], onerow[9], onerow[10], onerow[11], onerow[12], onerow[13],
         onerow[14], onerow[15], onerow[16], onerow[17], onerow[18], onerow[19], onerow[20],
         onerow[21], onerow[22], onerow[23], onerow[24], onerow[25], onerow[26], onerow[27]] = struct.unpack('28B',string[index:index+28])
        index += 28
        oneimage.append(onerow)
    imagematrix = np.mat(oneimage)
    trainImage.append(imagematrix)
    if ((i+1)%10000 == 0):
        print("has read ",(i+1)," images in train file")

# 读入训练标签
f = open(trainlabelpath, 'rb')
string = f.read()
print('train label file size:',len(string))
index = 8
for i in range(trainSize):
    [label] = struct.unpack('B',string[index:index+1])
    index += 1
    trainLabel.append(label)

print(trainLabel)

# 读入测试样本
f = open(testimagepath, 'rb')
string = f.read()
print('test image file size:',len(string))
index=16
for i in range(testSize):
    oneimage = []
    for j in range(row):
        onerow = np.zeros(row)
        [onerow[0], onerow[1], onerow[2], onerow[3], onerow[4], onerow[5], onerow[6],
         onerow[7], onerow[8], onerow[9], onerow[10], onerow[11], onerow[12], onerow[13],
         onerow[14], onerow[15], onerow[16], onerow[17], onerow[18], onerow[19], onerow[20],
         onerow[21], onerow[22], onerow[23], onerow[24], onerow[25], onerow[26], onerow[27]] = struct.unpack('28B',string[index:index+28])
        index += 28
        oneimage.append(onerow)
    imagematrix = np.mat(oneimage)
    testImage.append(imagematrix)
    if ((i+1)%10000 == 0):
        print("has read ",(i+1)," images in test file")


# 读入测试标签
f = open(testlabelpath, 'rb')
string = f.read()
print('test label file size:',len(string))
index = 8
for i in range(testSize):
    [label] = struct.unpack('B',string[index:index+1])
    index += 1
    testLabel.append(label)

print(testLabel)
