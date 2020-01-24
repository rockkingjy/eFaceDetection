/*

 */

#ifndef FEATURES_HAARFEATURES_H_
#define FEATURES_HAARFEATURES_H_

#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>
#include <iostream>
#include <vector>
#include "../classifiers/WeakClassifier.h"
#include "../utils/IntegralImage.h"

#define TOT_FEATURES 105106
//#define TOT_FEATURES 80960

using namespace cv;
using namespace std;

class HaarFeatures
{
public:
	static void getFeature(int size, WeakClassifier *wc);
	static vector<float> extractFeatures(Mat img, int size);
	static vector<float> extractFeatures(Mat integralImage, int size, bool store, WeakClassifier *wc);
	static float evaluate(Mat intImg, vector<Rect> whites, vector<Rect> blacks);
};

#endif /* FEATURES_HAARFEATURE_H_ */
