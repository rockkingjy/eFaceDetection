/*

 */

#ifndef FACEDETECTOR_H_
#define FACEDETECTOR_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <ctime>
#include <sstream>
#include "features/Data.h"
#include "features/HaarFeatures.h"
#include "utils/IntegralImage.h"
#include "ViolaJones.h"
#include "utils/Face.h"
#include "utils/Utils.hpp"

using namespace std;
using namespace cv;

class FaceDetector {

private:
	string positivePath;
	string negativePath;
	string validationPath;
	int numPositives;
	int numNegatives;
	int numValidation;

	int detectionWindowSize;
	int scales;
	int stages;
	float delta;
	vector<Mat> scaledImages;
	ViolaJones* boost;

public:
	FaceDetector(string trainedCascade, int scales = 12);
	FaceDetector(string positivePath, string negativePath, int stages, int numPositives, int numNegatives, int detectionWindowSize = 24);
	~FaceDetector();

	void setValidationSet(string validationPath, int examples = 0);
	void train();
	vector<Face> detect(Mat img, bool showResults = false, bool showScores = false);
	void displaySelectedFeatures(Mat img, int index = -1, string save_path = "");
	
};



#endif /* FACEDETECTOR_H_ */
