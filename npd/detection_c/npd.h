
#pragma once

#include <stdio.h>

#include "common.h"

#define DETECTION_MAX 100
static void *cascade = NULL;
static float rcsq[4 * DETECTION_MAX];

int face_detection(image* im, float* rcsq, void *cascade, int *lpoints, int *rpoints);