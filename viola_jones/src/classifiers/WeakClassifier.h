/*

 */

#ifndef WEAKCLASSIFIER_H_
#define WEAKCLASSIFIER_H_

#include <iostream>
#include <vector>
#include "../features/Data.h"
#include "../utils/Utils.hpp"

using namespace std;
using namespace cv;

class WeakClassifier
{

private:
	int dimension; // dimension of the feature chosed for this weakclassifier
	float threshold;
	float alpha;
	float beta;
	example sign; // sign for weekclassifier = pj in paper
	float error;
	int misclassified;

	//ViolaJones attributes
	vector<Rect> whites;
	vector<Rect> blacks;

public:
	WeakClassifier();
	~WeakClassifier() {}

	int predict(Data *x);
	int predict(float value);
	int predict(const vector<float> &x);
	void evaluateError(vector<Data *> &features);

	void printInfo();
	float getError() const;
	void setError(float error);
	int getDimension() const;
	void setDimension(int dimension);
	float getThreshold() const;
	void setThreshold(float threshold);
	float getAlpha() const;
	void setAlpha(float alpha);
	float getBeta() const;
	void setBeta(float beta);
	example getSign() const;
	void setSign(example sign);
	int getMisclassified() const;
	void setMisclassified(int misclassified);
	const vector<Rect> &getBlacks() const;
	void setBlacks(const vector<Rect> &blacks);
	const vector<Rect> &getWhites() const;
	void setWhites(const vector<Rect> &whites);
};

#endif /* BOOSTING_CLASSIFIERS_WEAKCLASSIFIER_H_ */
