package kNN;

/**
 * 包含各种距离准则的类
 * 
 * @author chuzhumin
 *
 */
public class Distance {
	/**
	 * 前两个参数为需要计算距离的样本点
	 * p为p-范数，这里p<=150，太大的话会溢出
	 * 
	 * @param sample1
	 * @param sample2
	 * @param p
	 * @return
	 */
	public static double distanceNorm(double[][] sample1, double[][] sample2, double p) {
		double dist = 0.0d;
		for (int i = 0; i < sample1.length; i++) {
			for (int j = 0; j < sample1[0].length; j++) {
				double absDelta = Math.abs(sample1[i][j] - sample2[i][j]);
				dist += Math.pow(absDelta, p);
			}
		}
		dist = (double) Math.pow(dist, 1/p);
		return dist;
	}
	
	/**
	 * 采用Chebychev距离计算距离
	 * 
	 * @param sample1
	 * @param sample2
	 * @return
	 */
	public static double distanceChebychev(double[][] sample1, double[][] sample2) {
		double dist = 0.0d;
		for (int i = 0; i < sample1.length; i++) {
			for (int j = 0; j < sample1[0].length; j++) {
				double absDelta = Math.abs(sample1[i][j] - sample2[i][j]);
				if (absDelta > dist) {
					dist = absDelta;
				}
			}
		}
		return dist;
	}
}
