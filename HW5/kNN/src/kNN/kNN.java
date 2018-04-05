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
		
	}
}
