package com.demo.facerecognition;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;

public class FaceRecognitionEigenFaces {
	public static String basePath=System.getProperty("user.dir");

	public static String csvFilePath=basePath+"\\src\\resources\\FaceRecognition\\TrainingData.txt";
	public static void main(String[] args) {
		System.out.println("Loading library..");
		System.loadLibrary("libopencv_java342");
		System.out.println("Library loaded!!");
		ArrayList<Mat> images=new ArrayList<>();
		ArrayList<Integer> labels=new ArrayList<>();
		readCSV(csvFilePath,images,labels);
		
		Mat testSample=images.get(images.size()-1);
		Integer testLabel=labels.get(images.size()-1);
		images.remove(images.size()-1);
		labels.remove(labels.size()-1);
		MatOfInt labelsMat=new MatOfInt();
		labelsMat.fromList(labels);
		EigenFaceRecognizer efr=EigenFaceRecognizer.create();
		System.out.println("Starting training...");
		efr.train(images, labelsMat);
		
		int[] outLabel=new int[1];
		double[] outConf=new double[1];
		System.out.println("Starting Prediction...");
		 efr.predict(testSample,outLabel,outConf);
		 
		System.out.println("***Predicted label is "+outLabel[0]+".***");

		System.out.println("***Actual label is "+testLabel+".***");
		System.out.println("***Confidence value is "+outConf[0]+".***");

	}
	private static void readCSV(String csvFilePath2, ArrayList<Mat> images, ArrayList<Integer> labels)  {
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(csvFilePath2));
		
		String line;
		while((line=br.readLine())!=null){
			String[] tokens=line.split("\\;");
			Mat readImage=Imgcodecs.imread(tokens[0], 0);
			images.add(readImage);
			labels.add(Integer.parseInt(tokens[1]));
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
