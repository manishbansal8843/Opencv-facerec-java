package com.demo;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

public class HelloWorld {

	public static void main(String[] args) {
		System.loadLibrary("libopencv_java342");
        Mat mat = Mat.eye(3, 3, CvType.CV_8UC1);
        System.out.println("mat = " + mat.dump());
	}

}
