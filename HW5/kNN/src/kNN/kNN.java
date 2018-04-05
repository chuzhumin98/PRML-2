package kNN;

public class kNN {
	public ReadData trainSet; //训练集数据 
	public ReadData testSet; //测试集数据
	
	/**
	 * 生成训练测试集，并导入数据
	 */
	public kNN() {
		trainSet = new ReadData();
		this.testSet = new ReadData();
		this.trainSet.readData(true);
		this.testSet.readData(false);
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
