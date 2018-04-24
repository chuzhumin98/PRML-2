package kNN;

import java.util.ArrayList;
import java.util.Collections;

public class kNN {
	public ReadData trainSet; //训练集数据 
	public ReadData testSet; //测试集数据
	
	public int usedTrainSize = 60000; //使用的训练集大小
	public int usedTestSize = 10; //使用的测试集大小
	
	ArrayList<Integer> trainIndex = new ArrayList<Integer>(); //训练集的随机抽样数组
	ArrayList<Integer> testIndex = new ArrayList<Integer>(); //测试集的随机抽样数组
	
	/**
	 * 生成训练测试集，并导入数据，同时随机抽取需要的数据规模大小的数据
	 */
	public kNN() {
		trainSet = new ReadData();
		this.testSet = new ReadData();
		this.trainSet.readData(true);
		this.testSet.readData(false);
		for (int i = 0; i < this.trainSet.size; i++) {
			this.trainIndex.add(i);
		}
		for (int i = 0;i < this.testSet.size; i++) {
			this.testIndex.add(i);
		}
		this.shuffleData();
	}
	
	/**
	 * 对训练集和测试集的数据进行随机打乱
	 */
	public void shuffleData() {
		Collections.shuffle(trainIndex); //采用shuffle方法对数据集进行打乱
		Collections.shuffle(testIndex);
	}
	
	/**
	 * 进行最近邻规则分类，输出[正确率，时间消耗]
	 * 参数p表示p范数，当p>150时采用Chebychev距离
	 * 
	 * @param p
	 * @return
	 */
	public double[] doMNN(int p) {
		long time1 = System.currentTimeMillis(); //记录当前时间戳
		int countRight = 0; //记录总的打对了的标签
		for (int i = 0; i < this.usedTestSize; i++) {
			int maxIndex = -1; //最近邻的索引值
			double minDist = Double.MAX_VALUE; //最近邻的距离
			for (int j = 0; j < this.usedTrainSize; j++) {
				double myDist = 0.0;
				if (p > 150) {
					myDist = Distance.distanceChebychev(this.trainSet.images[this.trainIndex.get(j)], 
							this.testSet.images[this.testIndex.get(i)]);
				} else {
					myDist = Distance.distanceNorm(this.trainSet.images[this.trainIndex.get(j)], 
							this.testSet.images[this.testIndex.get(i)], p);
				}
				if (myDist < minDist) {
					maxIndex = j;
					minDist = myDist;
				}
			}
			if (this.testSet.labels[this.testIndex.get(i)] 
					== this.trainSet.labels[this.trainIndex.get(maxIndex)]) { //判断是否打对标签
				countRight++;
			}
			System.out.println("no "+(i+1)+": real_"+this.testSet.labels[this.testIndex.get(i)]
		              +" test_"+this.trainSet.labels[this.trainIndex.get(maxIndex)]);
		}
		double accuracy = 1.0*countRight/this.usedTestSize; //计算准确率
		long time2 = System.currentTimeMillis();
		double costTime = (double)(time2-time1);
		System.out.println("accuracy:"+accuracy);
		System.out.println("cost time:"+costTime+" ms");
		double[] out = {accuracy, costTime};
		return out;
	}
	
	
	
	public static void main(String[] args) {
		kNN knn = new kNN();
		knn.doMNN(1);
	}
}
