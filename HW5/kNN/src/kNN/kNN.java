package kNN;

import java.util.ArrayList;
import java.util.Collections;

public class kNN {
	public ReadData trainSet; //训练集数据 
	public ReadData testSet; //测试集数据
	
	public int usedTrainSize = 5000; //使用的训练集大小
	public int usedTestSize = 1000; //使用的测试集大小
	
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
		int maxIndex = -1; //最近邻的索引值
		double minDist = Double.MAX_VALUE; //最近邻的距离
		return null;
	}
	
	
	
	public static void main(String[] args) {
		kNN knn = new kNN();
		System.out.println(Double.MAX_VALUE);
		System.out.println(Distance.distanceNorm(knn.trainSet.images[0], knn.testSet.images[0], 1));
		System.out.println(Distance.distanceNorm(knn.trainSet.images[0], knn.testSet.images[0], 2));
		System.out.println(Distance.distanceNorm(knn.trainSet.images[0], knn.testSet.images[0], 3));
		System.out.println(Distance.distanceNorm(knn.trainSet.images[0], knn.testSet.images[0], 19));
		System.out.println(Distance.distanceNorm(knn.trainSet.images[0], knn.testSet.images[0], 100));
		System.out.println(Distance.distanceChebychev(knn.trainSet.images[0], knn.testSet.images[0]));
	}
}
