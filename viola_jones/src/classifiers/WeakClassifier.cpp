/*

 */

#include "WeakClassifier.h"

WeakClassifier::WeakClassifier() : error(1.), dimension(0),
								   threshold(0.), alpha(0.), beta(0.),
								   sign(POSITIVE), misclassified(0) {}

/**
 * Predict feature label
 */
int WeakClassifier::predict(Data *x)
{
	return predict(x->getFeatures()[dimension]);
}

int WeakClassifier::predict(const vector<float> &x)
{
	return predict(x[dimension]);
}

int WeakClassifier::predict(float value)
{
	if (value <= threshold)
	{
		if (sign == POSITIVE)
			return 1;
		else
			return -1;
	}
	else
	{
		if (sign == POSITIVE)
			return -1;
		else
			return 1;
	}
}

/**
 * Evaluate weighted error base on weights and misclassified samples
 */
void WeakClassifier::evaluateError(vector<Data *> &features)
{
	error = 0;
	misclassified = 0;
	for (int i = 0; i < features.size(); ++i)
	{
		int pred = predict(features[i]);
		if (pred != features[i]->getLabel())
		{
			error += features[i]->getWeight();
			misclassified += 1;
		}
	}
}

// ----
void WeakClassifier::printInfo()
{
	std::cout << "\nTrained WeakClassifier: alpha: " << alpha
			  << ", dimension: " << dimension
			  << ", error: " << error
			  << ", misclassified: " << misclassified
			  << ", threshold: " << threshold
			  << ", sign: ";
	if (sign == POSITIVE)
	{
		std::cout << "positive" << std::endl;
	}
	else
	{
		std::cout << "negative" << std::endl;
	}
}

float WeakClassifier::getError() const
{
	return error;
}

void WeakClassifier::setError(float error)
{
	this->error = error;
}

int WeakClassifier::getDimension() const
{
	return dimension;
}

void WeakClassifier::setDimension(int dimension)
{
	this->dimension = dimension;
}

float WeakClassifier::getThreshold() const
{
	return threshold;
}

void WeakClassifier::setThreshold(float threshold)
{
	this->threshold = threshold;
}

float WeakClassifier::getAlpha() const
{
	return alpha;
}

void WeakClassifier::setAlpha(float alpha)
{
	this->alpha = alpha;
	//this->beta = exp(-alpha); // yan
}

float WeakClassifier::getBeta() const
{
	return beta;
}

void WeakClassifier::setBeta(float beta)
{
	this->beta = beta;
}

example WeakClassifier::getSign() const
{
	return sign;
}

void WeakClassifier::setSign(example sign)
{
	this->sign = sign;
}

int WeakClassifier::getMisclassified() const
{
	return misclassified;
}

void WeakClassifier::setMisclassified(int misclassified)
{
	this->misclassified = misclassified;
}

const vector<Rect> &WeakClassifier::getBlacks() const
{
	return blacks;
}

void WeakClassifier::setBlacks(const vector<Rect> &blacks)
{
	this->blacks = blacks;
}

const vector<Rect> &WeakClassifier::getWhites() const
{
	return whites;
}

void WeakClassifier::setWhites(const vector<Rect> &whites)
{
	this->whites = whites;
}

