/*

 */

#include "Data.h"

Data::Data(vector<float> features) : features(features),
									 label(0),
									 weight(0.) {}

Data::Data(vector<float> features, int label) : features(features),
												label(label),
												weight(0.) {}

Data::~Data()
{
	features.clear();
}

// ---
void Data::print()
{
	cout << "[";
	for (int i = 0; i < features.size(); ++i)
	{
		cout << features[i];
		if (i < features.size() - 1)
		{
			cout << ", ";
		}
	}
	cout << "] (label: " << label << ", weight: " << weight << ")" << endl;
}

const vector<float> &Data::getFeatures() const
{
	return features;
}

void Data::setFeatures(const vector<float> &features)
{
	this->features = features;
}

int Data::getLabel() const
{
	return label;
}

void Data::setLabel(int label)
{
	this->label = label;
}

float Data::getWeight() const
{
	return weight;
}

void Data::setWeight(float weight)
{
	this->weight = weight;
}
