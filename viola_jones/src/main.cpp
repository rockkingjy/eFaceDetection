/*
 * Main.cpp

 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <exception>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "AdaBoost.h"
#include "features/Data.h"
#include "features/HaarFeatures.h"
#include "utils/IntegralImage.h"
#include "utils/Utils.hpp"
#include "FaceDetector.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv ){

	string imagePath = "/home/neusoft/amy/eFaceDetection/";

	//Utils::generateNonFacesDataset(imagePath + "backgrounds/", imagePath + "validation", 5000, 24);
	string positivePath = imagePath + "lfwcrop/faces/";
	string negativePath = imagePath + "backgrounds/";
	//string validationPath = imagePath + "validation/";


	Mat test = imread(imagePath + "imgs/ex_face_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat test = imread(imagePath + "test/knex0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat test = imread(imagePath + "lfwcrop/faces/Ana_Isabel_Sanchez_0001.pgm", CV_LOAD_IMAGE_GRAYSCALE);


	FaceDetector* detector = new FaceDetector(positivePath, negativePath, 12, 3000, 3000);
	//etector->setValidationSet(validationPath);
	detector->train();

	FaceDetector* detector = new FaceDetector("../weights/trainedDataBest.txt", 1);
	string save_path = "/home/neusoft/amy/eFaceDetection/viola_jones/tmp/";
	detector->displaySelectedFeatures(test, -1, save_path);
	detector->detect(test, true);

	/*
	string digitsPath = imagePath + "digits/train-images-idx3-ubyte";
	string digitsLabelsPath = imagePath + "digits/train-labels-idx1-ubyte";
	DigitsClassifier* digitsClassifier = new DigitsClassifier(digitsPath, digitsLabelsPath, 100);
	digitsClassifier->train();
	*/

    return 0;
}
