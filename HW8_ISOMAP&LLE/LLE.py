import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 计算W矩阵，取k个近邻
def calculateWeightMatrix(data, k):
    size = len(data)
    onevector = np.ones([k,1],np.float64)
    W = np.zeros([size,size],np.float64) #权重矩阵
    for i in range(size):
        distance = np.sqrt(np.sum(np.square(np.tile(data[i, :], [size, 1]) - data), axis=1))
        distance[i] = np.float('inf') #将自己的距离设为无穷大
        indexs = np.argpartition(distance, k-1)[0:k]
        delta = data[indexs,:]-np.tile(data[i,:], [k,1]) #k*d矩阵
        Z = np.matmul(delta, np.transpose(delta))
        Zni = np.linalg.inv(Z) #Z的逆
        numerator = np.matmul(Zni,onevector) #分子项
        denerator = np.matmul(np.transpose(onevector), numerator) #分母归一化项
        weightArray =  numerator[:,0] / denerator[0,0] #需要取出其中的每一行元素进行计算,len=k
        W[i,indexs] = weightArray #将这些位置赋上权重值
    return W

# 在d维空间中去最小化M矩阵的误差
def minimizeCost(M, d):
    print(M)
    Lambda, V = np.linalg.eig(M)  # 对M做矩阵特征值分解
    print(Lambda)

if __name__ == '__main__':
    # 设计实验数据，形状类似于"3"
    sizePart = 200 #一部分的样本点个数
    data = np.zeros([sizePart*2, 3], np.float64) #最后生成的数据
    # 第一部分：(Rsin\theta, y, Rcos\theta+2),1.5<R<2,0<y<1,0<\theta<pi
    paras = np.random.random((sizePart,3)) #各列分别是R,\theta,y
    data[:sizePart,0] = (paras[:,0]*0.5+1.5) * np.sin(paras[:,1]*np.pi)
    data[:sizePart,1] = paras[:,2]
    data[:sizePart, 2] = (paras[:, 0] * 0.5 + 1.5) * np.cos(paras[:, 1] * np.pi) + 2
    # 第二部分：(Rsin\theta, y, Rcos\theta-2),1.5<R<2,0<y<1,0<\theta<pi
    paras = np.random.random((sizePart, 3))  # 各列分别是R,\theta,y
    data[sizePart:sizePart*2, 0] = (paras[:, 0] * 0.5 + 1.5) * np.sin(paras[:, 1] * np.pi)
    data[sizePart:sizePart*2, 1] = paras[:, 2]
    data[sizePart:sizePart*2, 2] = (paras[:, 0] * 0.5 + 1.5) * np.cos(paras[:, 1] * np.pi) - 2
    # LLE主干代码
    W = calculateWeightMatrix(data, 20) #得到k近邻下的权值矩阵
    M = np.matmul(np.transpose(np.eye(len(W))-W), np.eye(len(W))-W)
    minimizeCost(M,2)

    """
    plt.figure(2)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(data[0:sizePart, 0], data[0:sizePart, 1], data[0:sizePart, 2], c='b')  # 绘制数据点
    ax.scatter(data[sizePart:sizePart * 2, 0], data[sizePart:sizePart * 2, 1], data[sizePart:sizePart * 2, 2],c='r')  # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title('initial data distribution')
    plt.savefig('data2.png', dpi=150)
    """
