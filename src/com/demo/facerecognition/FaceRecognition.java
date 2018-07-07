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

public class FaceRecognition {
	public static String basePath=System.getProperty("user.dir");

	public static String csvFilePath=basePath+"\\src\\resources\\at.txt";
	public static void main(String[] args) {
		System.loadLibrary("libopencv_java342");
		System.out.println("Library loaded..");
		ArrayList<Mat> images=new ArrayList<>();
		ArrayList<Integer> labels=new ArrayList<>();
		
		
			readCSV(csvFilePath,images,labels);
			
		//int height=images.get(0).rows();
		int size=images.size();
		StringBuilder sb=new StringBuilder();
		for(int i=0;i<size;i++){
			sb.append("("+images.get(i).rows()+","+images.get(i).cols()+"),");
		}
		
		//System.out.println("images:"+images);
		Mat testSample=images.get(images.size()-1);
		//Mat testGraySample=new Mat();
		Integer testLabel=labels.get(images.size()-1);
		images.remove(images.size()-1);
		labels.remove(labels.size()-1);
		MatOfInt labelsMat=new MatOfInt();
		labelsMat.fromList(labels);
		EigenFaceRecognizer efr=EigenFaceRecognizer.create();
		System.out.println("Starting training...");
		efr.train(images, labelsMat);
		//Imgproc.cvtColor(testSample, testGraySample, Imgproc.COLOR_BGR2GRAY);
		//Imgproc.equalizeHist(testGraySample, testGraySample);

		//efr.
		System.out.println("eigenvalues rows,cols: ("+efr.getEigenValues().rows()+","+efr.getEigenValues().cols()+")");
		System.out.println("eigenvector rows,cols: ("+efr.getEigenVectors().rows()+","+efr.getEigenVectors().cols()+"),type:"+efr.getEigenVectors().type());
		System.out.println("mean rows,cols: ("+efr.getMean().rows()+","+efr.getMean().cols()+"),mean total:"+efr.getMean().total());
		//System.out.println("Training finished. Time to predict...test dims,rows,cols:"+testSample.dims()+","+testSample.rows()+","+testSample.cols()+","+testSample.type());
		//System.out.println("Training finished. Time to predict...test dims,rows,cols:"+testGraySample.dims()+","+testGraySample.rows()+","+testGraySample.cols()+","+testGraySample.type());

		int[] outLabel=new int[1];
		double[] outConf=new double[1];
		 efr.predict(testSample,outLabel,outConf);
		 
		System.out.println("Predicted label is:"+outLabel[0]);
		System.out.println("Predict label is:"+efr.predict_label(testSample));

		System.out.println("Actual label is:"+testLabel);
		System.out.println("predicted conf:"+outConf[0]);

	}
	private static void readCSV(String csvFilePath2, ArrayList<Mat> images, ArrayList<Integer> labels)  {
		BufferedReader br;
		try {
			br = new BufferedReader(new FileReader(csvFilePath));
		
		String line;
		while((line=br.readLine())!=null){
			String[] tokens=line.split("\\;");
			//System.out.println("csv:"+tokens[0]+","+tokens[1]);
			//File grayOutputfile = new File(tokens[0].substring(0,tokens[0].length()-4)+"_mod.jpg");

			Mat readImage=Imgcodecs.imread(tokens[0], 0);
			//Utils.matToBufferedImage(readImage);
		   // ImageIO.write(Utils.matToBufferedImage(readImage), "jpg", grayOutputfile);

			images.add(readImage);
			labels.add(Integer.parseInt(tokens[1]));
		}
		//System.out.println("images all converted");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

}
