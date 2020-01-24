/*

 */

#ifndef CASCADECLASSIFIER_H_
#define CASCADECLASSIFIER_H_

#include <vector>
#include "Stage.h"

using namespace std;

class CascadeClassifier {
private:
	vector<Stage*> stages;

public:
	CascadeClassifier();
	~CascadeClassifier(){};

	void addStage(Stage* stage);
	int predict(Mat img);
	int predict(const vector<float>& x);
	float score(const vector<float>& x);//no imp
	void train();//no imp

	const vector<Stage*>& getStages() const;
	void setStages(const vector<Stage*>& stages);
};

#endif /* CASCADECLASSIFIER_H_ */
