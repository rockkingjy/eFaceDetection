/*

 */

#ifndef STRONGCLASSIFIER_H_
#define STRONGCLASSIFIER_H_

#include <vector>
#include <iostream>
#include "WeakClassifier.h"
#include "../features/Data.h"

class StrongClassifier
{
private:
	float threshold;

protected:
	vector<WeakClassifier *> classifiers;

public:
	StrongClassifier(vector<WeakClassifier *> classifiers);
	~StrongClassifier();

	int predict(Data *x);

	float getThreshold() const;
	void setThreshold(float threshold);
	const vector<WeakClassifier *> &getClassifiers() const;
	void setClassifiers(const vector<WeakClassifier *> &classifiers);
};

#endif /* STRONGCLASSIFIER_H_ */
