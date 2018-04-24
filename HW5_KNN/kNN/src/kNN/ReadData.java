package kNN;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;

public class ReadData {
	public double[][][] images; //存储图像
	public int[] labels; //存储标签
	public static final String trainImages = "input/train_images";
	public static final String testImages = "input/test_images";
	public static final String trainLabels = "input/train_labels";
	public static final String testLabels = "input/test_labels";
	public int size; //数据集的规模
	public int row; //样本点的行数
	public int column; //样本点的列数
	
	/**
	 * 读取图像信息
	 * 
	 * @param path
	 */
	private void readImages(String path) {
		try {
			DataInputStream input = new DataInputStream(  
			        new BufferedInputStream(  
			        new FileInputStream(path)));
			int magic = input.readInt();
			this.size = input.readInt();
			this.row = input.readInt();
			this.column = input.readInt();
			this.images = new double [this.size][this.row][this.column];
			for (int i = 0; i < this.size; i++) {
				for (int j = 0; j < this.row; j++) {
					for (int k = 0; k < this.column; k++) {
						this.images[i][j][k] = (double)(input.readUnsignedByte());
					}
				}
				if ((i+1)%10000 == 0) {
					System.out.println("has read "+(i+1)+" images in "+magic);
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 		
	}
	
	/**
	 * 读取标签信息
	 * 
	 * @param path
	 */
	private void readLabels(String path) {
		try {
			DataInputStream input = new DataInputStream(  
			        new BufferedInputStream(  
			        new FileInputStream(path)));
			int magic = input.readInt();
			this.size = input.readInt();
			this.labels = new int [this.size];
			for (int i = 0; i < this.size; i++) {
				this.labels[i] = (Integer)(input.readUnsignedByte());
			}
			System.out.println("has read all the labels in "+magic);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 		
	}
	
	/**
	 * 读取图像数据和配套的标签数据
	 * 参数true为训练集，false为测试集
	 * 
	 * @param isTrainData
	 */
	public void readData(boolean isTrainData) {
		if (isTrainData) {
			this.readImages(trainImages);
			this.readLabels(trainLabels);
		} else {
			this.readImages(testImages);
			this.readLabels(testLabels);
		}
	}
	
	public static void main(String[] args) {
		ReadData rd = new ReadData();
		rd.readData(true);
	}
}
