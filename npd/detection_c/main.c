
#include "npd.h"

int main()
{

	char *model_name = "../weights/620.weight";
	char *filename = "../../imgs/smile.img";

	// read image
	int height = 272;//681;
	int width = 480;//1024;
	image im = load_image(filename, width, height, 1);
	debug("image read");
	uint8_t *pixels = (uint8_t *)im.data;

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

    // table for speed up
    int *lpoints = malloc(165600*sizeof(int));
	int *rpoints = malloc(165600*sizeof(int));
	size_t numPixels = 24 * 24;
	int points_ind = 0;
	for (int i = 0; i < numPixels; i++)
	{
		for (int j = i + 1; j < numPixels; j++)
		{
			lpoints[points_ind] = i;
			rpoints[points_ind] = j;
			points_ind++;
		}
	}
    debug("table made");

    clock_t before = clock();
    face_detection(&im, rcsq, cascade, lpoints, rpoints);
    clock_t difference = clock() - before;
    float msec = difference * 1000.0 / (float)CLOCKS_PER_SEC;
    debug("msec: %f ms", msec);

	free(cascade);
	free(lpoints);
	free(rpoints);
    return (0);
}
