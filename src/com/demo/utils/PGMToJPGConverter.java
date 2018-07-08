package com.demo.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class PGMToJPGConverter {
	public static String basePath=System.getProperty("user.dir");

	public static String csvFilePath=basePath+"\\src\\resources\\FaceRecognition\\TrainingData.txt";
	public static void main(String[] args) {
		System.out.println("Loading library..");
		System.loadLibrary("libopencv_java342");
		System.out.println("Library loaded!!");		
			readCSVAndConvertPGMToJPG(csvFilePath);
			System.out.println("Image conversion done!");
	}
	private static void readCSVAndConvertPGMToJPG(String csvFilePath2)  {
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(csvFilePath2));
		
		String line;
		while((line=br.readLine())!=null){
			String[] tokens=line.split("\\;");
			File grayOutputfile = new File(tokens[0].substring(0,tokens[0].length()-4)+".jpg");

			Mat readImage=Imgcodecs.imread(tokens[0], 0);
			Utils.matToBufferedImage(readImage);
		   ImageIO.write(Utils.matToBufferedImage(readImage), "jpg", grayOutputfile);

		}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}
