/*

 */

#ifndef FACE_H_
#define FACE_H_

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

class Face {

private:
	Rect rect;
	float score;
	bool evaluated;

public:
	Face(Rect rect);
	Face(Rect rect, float score);
	~Face();
	const Rect& getRect() const;
	void setRect(const Rect& rect);
	float getScore() const;
	bool isEvaluated() const;
	void setEvaluated(bool evaluated);
	void setScore(float score);
};

#endif /* FACE_H_ */

