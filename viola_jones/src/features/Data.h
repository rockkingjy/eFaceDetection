/*

 */

#ifndef FEATURE_DATA_H_
#define FEATURE_DATA_H_

#include <vector>
#include <string>
#include <iostream>

using namespace std;

class Data
{
private:
	vector<float> features;
	int label;		// ground truth label of this data
	float weight; 	// weight of this data

public:
	Data(vector<float> features);
	Data(vector<float> features, int label);
	~Data();

	void print();
	const vector<float> &getFeatures() const;
	void setFeatures(const vector<float> &features);
	int getLabel() const;
	void setLabel(int label);
	float getWeight() const;
	void setWeight(float weight);
};

#endif /* FEATURE_DATA_H_ */
