import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    # 设计实验数据，形状类似于"3"
    sizePart = 200 #一部分的样本点个数
    data = np.zeros([sizePart*2, 3], np.float32) #最后生成的数据
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

    plt.figure(2)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(data[0:sizePart, 0], data[0:sizePart, 1], data[0:sizePart, 2], c='b')  # 绘制数据点
    ax.scatter(data[sizePart:sizePart * 2, 0], data[sizePart:sizePart * 2, 1], data[sizePart:sizePart * 2, 2],c='r')  # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.title('initial data distribution')
    plt.show()
    plt.savefig('data2.png', dpi=150)
