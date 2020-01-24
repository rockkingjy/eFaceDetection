/*

 */

#include "IntegralImage.h"

using namespace std;
using namespace cv;

float IntegralImage::computeArea(Mat intImg, Rect r){
	float a1 = intImg.at<float>(r.y + r.height - 1, r.x + r.width - 1);
	float a2 = intImg.at<float>(r.y, r.x);
	float a3 = intImg.at<float>(r.y, r.x + r.width - 1);
	float a4 = intImg.at<float>(r.y + r.height - 1, r.x);
	return a1 + a2 - a3 - a4;
}

Mat IntegralImage::computeIntegralImage(Mat img){
	Mat output (img.rows, img.cols, CV_32F);
	Scalar intensity;
	float value;
	for(int r = 0; r < img.rows; ++r){
		for(int c = 0; c < img.cols; ++c){
			intensity = img.at<uchar>(r, c);
			value = (float) intensity[0];
			if(r == 0 && c == 0){
				output.at<float>(r, c) = value;
			} else if(r == 0 && c > 0){
				output.at<float>(r, c) = value + output.at<float>(r, c - 1);
			} else if(c == 0 && r > 0){
				output.at<float>(r, c) = value + output.at<float>(r - 1, c);
			} else {
				output.at<float>(r, c) = value + output.at<float>(r, c - 1)
						+ output.at<float>(r - 1, c) - output.at<float>(r - 1, c - 1);
			}
		}
	}
	return output;
}

Mat IntegralImage::computeIntegralSquaredImage(Mat img, float mean){
	Mat output (img.rows, img.cols, CV_32F);
	Scalar intensity;
	float value;
	for(int r = 0; r < img.rows; ++r){
		for(int c = 0; c < img.cols; ++c){
			intensity = img.at<uchar>(r, c);
			value = pow((((float) intensity[0]) - mean), 2);
			if(r == 0 && c == 0){
				output.at<float>(r, c) = value;
			} else if(r == 0 && c > 0){
				output.at<float>(r, c) = value + output.at<float>(r, c - 1);
			} else if(c == 0 && r > 0){
				output.at<float>(r, c) = value + output.at<float>(r - 1, c);
			} else {
				output.at<float>(r, c) = value + output.at<float>(r, c - 1)
						+ output.at<float>(r - 1, c) - output.at<float>(r - 1, c - 1);
			}
		}
	}
	return output;
}
