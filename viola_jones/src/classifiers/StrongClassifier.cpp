/*

 */

#include "StrongClassifier.h"

StrongClassifier::StrongClassifier(vector<WeakClassifier *> classifiers) : threshold(0.), classifiers(classifiers) {}

StrongClassifier::~StrongClassifier()
{
	classifiers.clear();
}

int StrongClassifier::predict(Data *x)
{
	float sum = 0;
	for (int i = 0; i < classifiers.size(); ++i)
	{
		sum += classifiers[i]->getAlpha() * classifiers[i]->predict(x);
	}
	if (sum > threshold)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

// ----
float StrongClassifier::getThreshold() const
{
	return threshold;
}

void StrongClassifier::setThreshold(float threshold)
{
	this->threshold = threshold;
}

const vector<WeakClassifier *> &StrongClassifier::getClassifiers() const
{
	return classifiers;
}

void StrongClassifier::setClassifiers(const vector<WeakClassifier *> &classifiers)
{
	this->classifiers = classifiers;
}
