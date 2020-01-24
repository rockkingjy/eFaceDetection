/*
 * IntegralImage.h
 *
 *  Created on: 21/mar/2016
 *      Author: lorenzocioni
 */

#ifndef INTEGRALIMAGE_H_
#define INTEGRALIMAGE_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;

class IntegralImage {
public:
	static Mat computeIntegralImage(Mat img);
	static Mat computeIntegralSquaredImage(Mat img, float mean);
	static float computeArea(Mat intImg, Rect r);
};

#endif /* INTEGRALIMAGE_H_ */
