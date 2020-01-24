/*

 */

#ifndef ADABOOST_H_
#define ADABOOST_H_

#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <ctime>

#include "classifiers/StrongClassifier.h"
#include "classifiers/WeakClassifier.h"

using namespace std;
using namespace chrono;

class AdaBoost
{

protected:
	int iterations; // iterrations of training
	vector<Data *> features;
	StrongClassifier *strongClassifier;

	WeakClassifier *trainWeakClassifier();

	virtual void updateWeights(WeakClassifier *weakClassifier);
	virtual void normalizeWeights();
	virtual float updateAlpha(float error);
	virtual float updateBeta(float error);

public:
	AdaBoost();
	AdaBoost(vector<Data *> &data, int iterations);
	virtual ~AdaBoost();
	
	int predict(Data *x);
	StrongClassifier *train();
	StrongClassifier *train(vector<WeakClassifier *> &classifiers);

	void showFeatures();
	int getIterations() const;
	void setIterations(int iterations);
};

#endif /* ADABOOST_H_ */
