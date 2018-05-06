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

    distanceMatrix = distance(data[:,:]) # 根据数据得到初始的距离矩阵
    manifoldMatrix = holdkNearest(distanceMatrix, 10) #得到top-k的初始流形距离
    for i in range(len(manifoldMatrix)):
        Dijkstra(manifoldMatrix,i)
    print(manifoldMatrix)

    """
    print(data)
    print(len(data))
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
    """