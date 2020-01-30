/*
 *  This code is released under the MIT License.
 *  Copyright (c) 2013 Nenad Markus
 */

#include "picornt_ex.h"


int face_detection()
{
	char *model_name = "./src/picornt.weight";
    char *filename = "./img/ex_face_3.img"; 
	
	// set default parameters
	void *cascade = 0;
	int minsize = 128;
	int maxsize = 1024;
	float angle = 0.0f;
	float scalefactor = 1.1f;
	float stridefactor = 0.1f;
	float qthreshold = 5.0f;
	int usepyr = 0;
	int noclustering = 0;

	// read weight
	int size;
	FILE *file;
	file = fopen(model_name, "rb");
	if (!file)
	{
		printf("# cannot read weight.\n");
		return 1;
	}
	fseek(file, 0L, SEEK_END);
	size = ftell(file);
	printf("weight size:%d\n", size);
	fseek(file, 0L, SEEK_SET);
	cascade = malloc(size);
	if (!cascade || size != fread(cascade, 1, size, file))
		return 1;
	fclose(file);
	debug("weight read");

	// read image
	int nrows = 856;
	int ncols = 1024;
	int ldim = 1024;
    in_out im = load_image(filename, ncols, nrows, 1);
	debug("image read");

	uint8_t *pixels = (uint8_t *)im.data;
	// do detection
#define MAXNDETECTIONS 100
	int ndetections;
	float rcsq[4 * MAXNDETECTIONS];

	ndetections = find_objects(rcsq, MAXNDETECTIONS, cascade, angle, pixels, 
								nrows, ncols, ldim, scalefactor, stridefactor, 
								minsize, nrows);
	debug("detections before clustering: %d", ndetections);

	if (!noclustering)
		ndetections = cluster_detections(rcsq, ndetections);

	debug("detections after clustering: %d", ndetections);
	for (int i = 0; i < ndetections; ++i)
		if (rcsq[4 * i + 3] >= qthreshold) // check the confidence threshold
			printf("%d %d %d %f\n", 
					(int)rcsq[4 * i + 0], // y, x, d, confidence
					(int)rcsq[4 * i + 1], 
					(int)rcsq[4 * i + 2], 
					rcsq[4 * i + 3]);

	return 0;
}
