import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入一个N*k的位置矩阵，返回一个N*N的距离矩阵
def distance(data):
    size = len(data)
    distMatrix = np.zeros([size,size],dtype=np.float32) #初始化距离矩阵
    for i in range(size):
        distMatrix[i,:] = np.sum(np.square(np.tile(data[i, :], [size, 1]) - data), axis=1)
    return distMatrix

# 只保留前k近邻的距离，其他均化为无穷——流形距离
def holdkNearest(distMatrix, k):
    size = len(distMatrix)
    manifoldDist = np.ones([size, size], np.float32)*np.float('inf') #流形距离
    for i in range(size):
        careArray = distMatrix[i,:] #在这里所关心的距离
        topk = np.argpartition(careArray, k)[0:k+1] #这个地方需要注意，topk其实是到前k+1个结果，因为自距离为0
        manifoldDist[i,topk] = careArray[topk] #将topk的距离替换成实际距离
        manifoldDist[topk,i] = careArray[topk]
    return manifoldDist

# 从start为index进行Dijkstra扩展
def Dijkstra(manifoldMatrix, start):
    cacheDist = manifoldMatrix[start].copy() #缓存一下距离
    cacheDist[start] = np.float32('inf') #将它本身的距离设为无穷，方便后面寻找距离最短的节点
    for i in range(len(manifoldMatrix)-1):
        index = np.argmin(cacheDist)
        dist = cacheDist[index] #记录一下该距离
        for j in range(len(manifoldMatrix)):
            # 如果有更小的距离选项时，则更新相关的距离
            if (manifoldMatrix[start][j] > dist + manifoldMatrix[index][j]):
                manifoldMatrix[start][j] = dist + manifoldMatrix[index][j]
                manifoldMatrix[j][start] = manifoldMatrix[start][j]
                cacheDist[j] = manifoldMatrix[start][j]
        cacheDist[index] = np.float('inf') #使用过的信息，距离归为无穷

# MDS实现数据降维，输入为距离矩阵(N*N)，输出为在p维中的位置N*p
def MDS(distMatrix, p):
    size = len(distMatrix)
    A = np.square(distMatrix) * (-0.5) #-d^2/2
    H = np.eye(size) - np.ones([size,size])/size #H矩阵
    B = np.matmul(H, A)
    B = np.matmul(B, H) #B=HAH
    Lambda, V = np.linalg.eig(B) #对B做矩阵特征值分解
    indexs = np.argpartition(-Lambda, p-1)[0:p] #取-Lambda最小的p个特征值，即为Lambda最大的p个
    Lambda1 = np.diag(Lambda[indexs]) #抽取的特征值组成的矩阵
    V1 = V[:, indexs]
    return np.matmul(V1, np.sqrt(Lambda1)) #得到降维后的数据

if __name__ == '__main__':
    # 在三维空间中生成形状为N的数据
    sizePart = 100 #一部分的点数
    # 第一部分：-3<x<-1,0<y<1,z=2x+4+(0-0.5)
    data = np.random.random((sizePart,3))
    data[:,0] = data[:,0]*2-3
    data[:,2] = data[:,0]*2+4+data[:,2]*0.5
    # 第二部分：-1<x<1,0<y<1,z=-2x+(0-0.5)
    dataTmp = np.random.random((sizePart,3))
    dataTmp[:,0] = dataTmp[:,0]*2-1
    dataTmp[:,2] = dataTmp[:,0]*(-2)+dataTmp[:,2]*0.5
    data = np.vstack((data,dataTmp)) #将前两部分数据合并
    # 第三部分：1<x<3,0<y<1,z=2x-4+(0-0.5)
    dataTmp = np.random.random((sizePart, 3))
    dataTmp[:, 0] = dataTmp[:, 0] * 2 + 1
    dataTmp[:, 2] = dataTmp[:, 0] * 2 - 4 + dataTmp[:, 2]*0.5
    data = np.vstack((data, dataTmp))  # 将三部分数据合并

    distanceMatrix = distance(data) # 根据数据得到初始的距离矩阵
    manifoldMatrix = holdkNearest(distanceMatrix, 10) #得到top-k的初始流形距离
    for i in range(len(manifoldMatrix)):
        Dijkstra(manifoldMatrix,i)
    newData = MDS(manifoldMatrix, 2) #得到ISOMAP后的数据

    plt.figure(1)
    plt.scatter(newData[0:sizePart, 0], newData[0:sizePart, 1], c='b')
    plt.scatter(newData[sizePart:sizePart * 2, 0], newData[sizePart:sizePart * 2, 1], c='g')
    plt.scatter(newData[sizePart * 2:sizePart * 3, 0], newData[sizePart * 2:sizePart * 3, 1], c='r')
    plt.xlabel('new feature 1')
    plt.ylabel('new feature 2')
    plt.title('data distribution after ISOMAP')
    plt.savefig('isomapdata1.png', dpi=150)

    plt.figure(2)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(data[0:sizePart,0], data[0:sizePart,1], data[0:sizePart,2], c='b')  # 绘制数据点
    ax.scatter(data[sizePart:sizePart*2, 0], data[sizePart:sizePart*2, 1], data[sizePart:sizePart*2, 2], c='g')  # 绘制数据点
    ax.scatter(data[sizePart*2:sizePart*3, 0], data[sizePart*2:sizePart*3, 1], data[sizePart*2:sizePart*3, 2], c='r')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title('initial data distribution')
    #plt.show()
    plt.savefig('data1.png',dpi=150)
