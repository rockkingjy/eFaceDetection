#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdint.h>

#define debug(a, args...) printf("%s(%s:%d) " a "\n", __func__, __FILE__, __LINE__, ##args)

typedef struct
{
    int8_t result;
    int8_t possibility;
} uinference_result;

typedef struct
{
    int h;
    int w;
    int c;
    int8_t *data;
} image;

typedef struct 
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float confidence;
    int label;
    float score;
} b_box;

image load_image(char *filename, int w, int h, int c);
void normalize_image(image *im, float mean, float std);
void normalize_image_255(image *im);
void free_in_out(image *m);
void print_in_out(image im);
void save_in_out(image im);

#endif
