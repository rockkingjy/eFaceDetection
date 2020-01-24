/*

 */

#include "AdaBoost.h"

AdaBoost::AdaBoost(vector<Data *> &data, int iterations) : iterations(iterations),
														   features(data),
														   strongClassifier(new StrongClassifier(vector<WeakClassifier *>{}))
{
	int size = features.size();
	cout << "Initializing AdaBoost with " << iterations << " iterations" << endl;
	cout << "Training size: " << size << "\n"
		 << endl;
	//Initialize weights
	for (int m = 0; m < features.size(); ++m)
	{
		features[m]->setWeight((float)1 / features.size());
	}
	cout << "Initialized uniform weights\n"
		 << endl;
}

AdaBoost::AdaBoost() : iterations(0),
					   strongClassifier(new StrongClassifier(vector<WeakClassifier *>{}))
{
}

AdaBoost::~AdaBoost()
{
	features.clear();
	cout << "Removing AdaBoost from memory" << endl;
}

int AdaBoost::predict(Data *x)
{
	return strongClassifier->predict(x);
}

StrongClassifier *AdaBoost::train()
{
	//The vector of weak classifiers
	vector<WeakClassifier *> classifiers;
	return train(classifiers);
}

/**
 * Train the AdaBoost classifier with a number of weak classifier specified
 * with the iteration attribute.
 */
StrongClassifier *AdaBoost::train(vector<WeakClassifier *> &classifiers)
{
	cout << "Training AdaBoost with " << iterations << " iterations" << endl;
	auto t_start = chrono::high_resolution_clock::now();

	//Iterate for the specified iterations
	if (classifiers.size() < iterations)
	{
		for (unsigned int i = classifiers.size(); i < iterations; ++i)
		{
			cout << "Iteration: " << (i + 1) << endl;
			WeakClassifier *weakClassifier = trainWeakClassifier(); // 1. train a week classifier
			float error = weakClassifier->getError();
			if (error < 0.5) // check the error is better than random guess or not
			{
				float alpha = updateAlpha(error);
				float beta = updateBeta(error);
				weakClassifier->setAlpha(alpha);
				weakClassifier->setBeta(beta);
				updateWeights(weakClassifier); // 2. updata weights
				normalizeWeights();			   // 3. normalize weights
				weakClassifier->printInfo();
				classifiers.push_back(weakClassifier); // push this week classifier into the stack
				if (error == 0)	//If error is 0, classification is perfect (linearly separable data)
				{
					break;
				}
			}
			else
			{
				cout << "Error: weak classifier with error > 0.5." << endl;
				break;
			}
		}
	}
	//showFeatures();
	strongClassifier->setClassifiers(classifiers); // set the stack as strong classifier

	auto t_end = high_resolution_clock::now();
	cout << "Training AdaBoost Time: " << (duration<double, milli>(t_end - t_start).count()) / 1000 << " s" << endl;
	return strongClassifier;
}

/***
 * Train weak classifier on training data choosing the one minimizing the error
 */
WeakClassifier *AdaBoost::trainWeakClassifier()
{
	WeakClassifier *bestWeakClass = new WeakClassifier();

	if (features.size() > 0)
	{
		//Feature vector dimension
		int dimensions = features[0]->getFeatures().size();
		//Error and signs vector
		vector<example> signs;
		vector<float> errors;
		vector<int> misclassifies;
		//Cumulative sums of the weights
		float posWeights = 0;	// S+
		float negWeights = 0;	// S-
		float totNegWeights = 0; // T+
		float totPosWeights = 0; // T-
		//Number of examples
		int totPositive = 0;
		int totNegative = 0;
		int cumPositive = 0;
		int cumNegative = 0;
		//Errors
		float weight, error;
		float errorPos, errorNeg;
		float threshold;
		int index;

		// 1. Evaluating total sum of negative and positive weights: T+, T-
		for (unsigned int i = 0; i < features.size(); ++i)
		{
			if (features[i]->getLabel() == 1)
			{
				totPosWeights += features[i]->getWeight();
				totPositive++;
			}
			else
			{
				totNegWeights += features[i]->getWeight();
				totNegative++;
			}
		}

		// Iterate through dimensions
		for (unsigned int j = 0; j < dimensions; ++j)
		{

			// 2. Sorts vector of features according to the j-th dimension
			sort(features.begin(), features.end(),
				 [j](Data *const &a, Data *const &b) { return a->getFeatures()[j] < b->getFeatures()[j]; });

			//Reinitialize variables
			signs.clear();
			errors.clear();
			misclassifies.clear();

			posWeights = 0;
			negWeights = 0;
			cumNegative = 0;
			cumPositive = 0;

			// 3. Iterates features
			for (int i = 0; i < features.size(); ++i)
			{
				// calculate S+, S-
				weight = features[i]->getWeight();
				if (features[i]->getLabel() == 1)
				{
					posWeights += weight;
					cumPositive++;
				}
				else
				{
					negWeights += weight;
					cumNegative++;
				}

				errorPos = posWeights + (totNegWeights - negWeights);
				errorNeg = negWeights + (totPosWeights - posWeights);

				if ((i < features.size() - 1 && features[i]->getFeatures()[j] != features[i + 1]->getFeatures()[j]) || i == features.size() - 1)
				{
					if (errorPos > errorNeg)
					{
						errors.push_back(errorNeg);
						signs.push_back(POSITIVE);
						misclassifies.push_back(cumNegative + (totPositive - cumPositive));
					}
					else
					{
						errors.push_back(errorPos);
						signs.push_back(NEGATIVE);
						misclassifies.push_back(cumPositive + (totNegative - cumNegative));
					}
				}
				else
				{
					errors.push_back(1.);
					signs.push_back(POSITIVE);
					misclassifies.push_back(0);
				}
			}

			// 4. get the min error, and the best week classifier till now
			auto errorMin = min_element(begin(errors), end(errors));
			error = *errorMin;
			if (error < bestWeakClass->getError())
			{
				index = errorMin - errors.begin();
				threshold = (features[index])->getFeatures()[j];
				bestWeakClass->setError(error);
				bestWeakClass->setDimension(j);
				bestWeakClass->setThreshold(threshold);
				bestWeakClass->setMisclassified(misclassifies[index]);
				bestWeakClass->setSign(signs[index]);
			}

			cout << "\rEvaluated: " << j + 1 << "/" << dimensions << " features" << flush;
		}
	}
	return bestWeakClass;
}

void AdaBoost::updateWeights(WeakClassifier *weakClassifier)
{
	for (int i = 0; i < features.size(); ++i)
	{
		float num = (features[i]->getWeight() *
					 exp(-weakClassifier->getAlpha() *
						 features[i]->getLabel() *
						 weakClassifier->predict(this->features[i])));
		features[i]->setWeight(num);
	}
}

void AdaBoost::normalizeWeights()
{
	float norm = 0;
	for (int i = 0; i < features.size(); ++i)
	{
		norm += features[i]->getWeight();
	}
	for (int i = 0; i < features.size(); ++i)
	{
		features[i]->setWeight((float)features[i]->getWeight() / norm);
	}
}

float AdaBoost::updateAlpha(float error)
{
	return log((1 - error) / error);
}

float AdaBoost::updateBeta(float error)
{
	return error / (1 - error);
}

void AdaBoost::showFeatures()
{
	for (int i = 0; i < features.size(); ++i)
	{
		features[i]->print();
	}
}

int AdaBoost::getIterations() const
{
	return iterations;
}

void AdaBoost::setIterations(int iterations)
{
	this->iterations = iterations;
}
