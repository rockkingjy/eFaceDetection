/*

 */

#include "CascadeClassifier.h"

CascadeClassifier::CascadeClassifier()
{
	this->stages = {};
}

void CascadeClassifier::addStage(Stage *stage)
{
	stages.push_back(stage);
}

int CascadeClassifier::predict(const vector<float> &x)
{
	for (int i = 0; i < stages.size(); ++i)
	{
		if (stages[i]->predict(x) != 1)
		{
			return 0;
		}
	}
	return 1;
}

int CascadeClassifier::predict(Mat img)
{
	for (int i = 0; i < stages.size(); ++i)
	{
		if (stages[i]->predict(img) != 1)
		{
			return 0;
		}
	}
	return 1;
}

const vector<Stage *> &CascadeClassifier::getStages() const
{
	return stages;
}

void CascadeClassifier::setStages(const vector<Stage *> &stages)
{
	this->stages = stages;
}

