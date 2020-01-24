/*

 */

#include "Face.h"

Face::Face(Rect rect): rect(rect), score(0.), evaluated(false){}

Face::Face(Rect rect, float score): rect(rect), score(score), evaluated(false){}

const Rect& Face::getRect() const {
	return rect;
}

void Face::setRect(const Rect& rect) {
	this->rect = rect;
}

bool Face::isEvaluated() const {
	return evaluated;
}

void Face::setScore(float score) {
	this->score = score;
}

void Face::setEvaluated(bool evaluated) {
	this->evaluated = evaluated;
}

float Face::getScore() const {
	return score;
}


Face::~Face() {
}
