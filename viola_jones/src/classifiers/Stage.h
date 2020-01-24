/*

 */

#ifndef STAGE_H_
#define STAGE_H_

#include <vector>
#include <iostream>
#include "WeakClassifier.h"
#include "../features/HaarFeatures.h"

using namespace std;

class Stage
{
private:
	int number;
	vector<WeakClassifier *> classifiers;
	float threshold;
	float fpr;
	float detectionRate;

public:
	Stage(int number);
	Stage(int number, vector<WeakClassifier *> weaks);
	~Stage();

	int predict(const vector<float> &x);
	int predict(Mat img);
	void optimizeThreshold(vector<Data *> &positiveSet, float dr);
	//void decreaseThreshold();

	void printInfo();
	float getThreshold() const;
	void setThreshold(float threshold);
	float getDetectionRate() const;
	void setDetectionRate(float detectionRate);
	float getFpr() const;
	void setFpr(float fpr);
	int getNumber() const;
	void setNumber(int number);
	const vector<WeakClassifier *> &getClassifiers() const;
	void setClassifiers(const vector<WeakClassifier *> &classifiers);
	void addClassifier(WeakClassifier *wc);
	
};

#endif /* STAGE_H_ */
