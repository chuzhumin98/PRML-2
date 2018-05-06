import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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