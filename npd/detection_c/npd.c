
#include "npd.h"

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

float get_overlap(float r1, float c1, float s1, float r2, float c2, float s2)
{
	float overr, overc;
	overr = max(0, min(r1 + s1 / 2, r2 + s2 / 2) - max(r1 - s1 / 2, r2 - s2 / 2));
	overc = max(0, min(c1 + s1 / 2, c2 + s2 / 2) - max(c1 - s1 / 2, c2 - s2 / 2));
	return overr * overc / (s1 * s1 + s2 * s2 - overr * overc);
}

void ccdfs(int a[], int i, float rcsq[], int n)
{
	for (int j = 0; j < n; ++j)
		if (a[j] == 0 && get_overlap(rcsq[4 * i + 0], rcsq[4 * i + 1], rcsq[4 * i + 2], rcsq[4 * j + 0], rcsq[4 * j + 1], rcsq[4 * j + 2]) > 0.5f)
		{
			a[j] = a[i];
			ccdfs(a, j, rcsq, n);
		}
}

int find_connected_components(int a[], float rcsq[], int n)
{
	int i, cc;
	if (!n)
		return 0;

	for (i = 0; i < n; ++i)
		a[i] = 0;

	cc = 1;
	for (i = 0; i < n; ++i)
		if (a[i] == 0)
		{
			a[i] = cc;
			ccdfs(a, i, rcsq, n);
			++cc;
		}
	return cc - 1; // number of connected components
}

int cluster_detections(float rcsq[], int n)
{
	int idx, ncc, cc;
	int a[4096];

	ncc = find_connected_components(a, rcsq, n);
	if (!ncc)
		return 0;
	idx = 0;

	for (cc = 1; cc <= ncc; ++cc)
	{
		int i, k;
		float sumqs = 0.0f, sumrs = 0.0f, sumcs = 0.0f, sumss = 0.0f;
		k = 0;
		for (i = 0; i < n; ++i)
			if (a[i] == cc)
			{
				sumrs += rcsq[4 * i + 0];
				sumcs += rcsq[4 * i + 1];
				sumss += rcsq[4 * i + 2];
				sumqs += rcsq[4 * i + 3];
				++k;
			}
		// average the connected detections.
		rcsq[4 * idx + 0] = sumrs / k;
		rcsq[4 * idx + 1] = sumcs / k;
		rcsq[4 * idx + 2] = sumss / k;
		rcsq[4 * idx + 3] = sumqs; // accumulated confidence measure
		++idx;
	}
	return idx;
}

int face_detection(image* im, float* rcsq, void *cascade, int *lpoints, int *rpoints)
{
	// set default parameters
	int ndetections = 0;
	uint8_t detection_flag = 0;
	int pWinSize[] = {24, 29, 35, 41, 50, 60, 72, 86, 103, 124, 149, 178, 214, 257, 308, 370, 444, 532, 639, 767, 920, 1104, 1325, 1590, 1908, 2290, 2747, 3297, 3956};
	int minFace = 20;
	int maxFace = 3000;
	int width = im->w;
	int height = im->h;
	int sizePatch = min(width, height);
	int thresh = 0;
	const uint8_t *O = (uint8_t *)im->data;
	unsigned char *I = malloc(width * height);
	// copy O to I
	int k = 0;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			I[k] = *(O + j * width + i);
			k++;
		}
	}
	free_in_out(im);
	// find the max pWinSize smaller than input img
	for (int i = 0; i < 8; i++)
	{
		if (sizePatch >= pWinSize[i])
		{
			thresh = i;
			continue;
		}
		else
		{
			break;
		}
	}
	thresh = thresh + 1; // the total no. of scales that will be searched
	minFace = max(minFace, 24);
	maxFace = min(maxFace, min(height, width));

	// read parameters
	int DetectSize = ((int *)cascade)[0];
	int stages = ((int *)cascade)[1];
	int numBranchNodes = ((int *)cascade)[2];
	debug("DetectSize: %d, stages: %d, numBranchNodes: %d", DetectSize, stages, numBranchNodes);

	int *treeIndex = (int *)(cascade + 12);
	float *thresholds = (float *)(cascade + 12 + stages * 4);
	int *feaIds = (int *)(cascade + 12 + stages * 8);
	int *leftChilds = (int *)(cascade + 12 + stages * 8 + numBranchNodes * 4);
	int *rightChilds = (int *)(cascade + 12 + stages * 8 + numBranchNodes * 8);
	uint8_t *cutpoints = (uint8_t *)(cascade + 12 + stages * 8 + numBranchNodes * 12);
	float *fits = (float *)(cascade + 12 + stages * 8 + numBranchNodes * 14);
	debug("Parameter load");
	
	printf("treeIndex:");
	for (int i = 0; i < stages; i++)
	{
		//printf("%d,", treeIndex[i]);
	}
	printf("\nthresholds:");
	for (int i = 0; i < stages; i++)
	{
		//printf("%f,", thresholds[i]);
	}
	printf("\n");
	debug("treeIndex: %d", *treeIndex);
	debug("thresholds: %f", *thresholds);
	debug("feaIds: %d", *feaIds);
	debug("leftChilds: %d", *leftChilds);
	debug("rightChilds: %d", *leftChilds);
	printf("cutpoints: %d\n", cutpoints[0]);
	printf("fits: %f\n", fits[0]);
	//exit(0);

	debug("thresh: %d", thresh);
	for (int k = 0; k < thresh; k++) // process each scale
	{
		debug("k: %d, ndetections: %d", k, ndetections);
		float factor = (float)pWinSize[k] / (float)DetectSize;

		if (pWinSize[k] < minFace)
			continue;
		else if (pWinSize[k] > maxFace)
			break;

		int winStep = (int)floor(pWinSize[k] * 0.1);
		if (pWinSize[k] > 40)
			winStep = (int)floor(pWinSize[k] * 0.05);

		int offset[pWinSize[k] * pWinSize[k]];
		int pp1 = 0, pp2 = 0, gap = height - pWinSize[k];
		for (int j = 0; j < pWinSize[k]; j++) // column coordinate
		{
			for (int i = 0; i < pWinSize[k]; i++) // row coordinate
			{
				offset[pp1++] = pp2++;
			}

			pp2 += gap;
		}
		int colMax = width - pWinSize[k] + 1;
		int rowMax = height - pWinSize[k] + 1;

		//debug("colMax: %d, rowMax: %d, winStep: %d", colMax, rowMax, winStep);
		for (int c = 0; c < colMax; c += winStep) // slide in column
		{
			const unsigned char *pPixel = I + c * height;

			for (int r = 0; r < rowMax; r += winStep, pPixel += winStep) // slide in row
			{
				float _score = 0;
				int s;

				// test each tree classifier
				for (s = 0; s < stages; s++)
				{
					int node = treeIndex[s];
					//debug("node: %d", node);
					// test the current tree classifier
					while (node > -1) // branch node
					{
						int feaid = feaIds[node];
						/*
						int tmp = 24 * 24 - 1;
						int res = feaid + 1;
						while (res > 0)
						{
							res = res - tmp;
							tmp--;
						}
						int lpoint = 24 * 24 - 1 - tmp - 1; //lpoints[feaid];
						int rpoint = 24 * 24 - 1 + res;		//rpoints[feaid];
						*/
						int lpoint = lpoints[feaid];
						int rpoint = rpoints[feaid];
						int p1x = lpoint / 24 * factor;
						int p1y = lpoint % 24 * factor;
						int p2x = rpoint / 24 * factor;
						int p2y = rpoint % 24 * factor;
						unsigned char p1 = pPixel[offset[p1y * pWinSize[k] + p1x]]; //pPixel[offset[points1[k][node]]];
						unsigned char p2 = pPixel[offset[p2y * pWinSize[k] + p2x]]; //pPixel[offset[points2[k][node]]];

						double fea = 0.5;
						if (p1 > 0 || p2 > 0)
						{
							fea = (double)p1 / ((double)p1 + (double)p2);
						}
						fea = floor(256 * fea);
						if (fea > 255)
						{
							fea = 255;
						}
						//unsigned char fea = ppNpdTable.at<uchar>(p1, p2);
						//debug("feaid, tmp, res:%d, %d, %d", feaid, tmp, res);
						//debug("lpoint: %d, rpoint: %d", lpoint, rpoint);
						/*
						debug("offset1, offset2: %ld, %ld", offset[p1y * pWinSize[k] + p1x], offset[p2y * pWinSize[k] + p2x]);
						debug("p1, p2, fea: %d, %d, %d", p1, p2, (int)fea);
						debug("cutpoints[2 * node]: %d", cutpoints[2 * node]);
						debug("cutpoints[2 * node + 1]: %d", cutpoints[2 * node + 1]);
						exit(0);*/
						if (fea < cutpoints[2 * node] || fea > cutpoints[2 * node + 1])
							node = leftChilds[node];
						else
							node = rightChilds[node];
					}

					node = -node - 1;
					_score = _score + fits[node];

					if (_score < thresholds[s])
					{
						break; // negative samples
					}
				}

				if (s == stages) // a face detected
				{
					/*
					bbox[detection_number].xmin = c;
					bbox[detection_number].ymin = r;
					bbox[detection_number].xmax = c + pWinSize[k];
					bbox[detection_number].ymax = r + pWinSize[k];
					bbox[detection_number].score = _score;
					*/
					rcsq[4 * ndetections] = r + pWinSize[k] / 2;
					rcsq[4 * ndetections + 1] = c + pWinSize[k] / 2;
					rcsq[4 * ndetections + 2] = pWinSize[k];
					rcsq[4 * ndetections + 3] = _score;
					//Rect roi(c, r, pWinSize[k], pWinSize[k]);
					//rects.push_back(roi);
					//scores.push_back(_score);
					ndetections++;

					if (ndetections >= DETECTION_MAX)
					{
						detection_flag = 1;
						break;
					}
				}
			}
			if (detection_flag == 1)
			{
				break;
			}
		}
		if (detection_flag == 1)
		{
			break;
		}
	}
	free(I);

	for (int i = 0; i < ndetections; i++)
	{
		//printf("%d, %d, %d, %d\n", (int)bbox[i].xmin, (int)bbox[i].ymin, (int)bbox[i].xmax, (int)bbox[i].ymax);
		printf("%f, %f, %f, %f\n", rcsq[i*4], rcsq[i*4+1], rcsq[i*4+2], rcsq[i*4+3]);
	}

	ndetections = cluster_detections(rcsq, ndetections);
	debug("ndetections:%d",ndetections);

	for (int i = 0; i < ndetections; i++)
	{
		//printf("%d, %d, %d, %d\n", (int)bbox[i].xmin, (int)bbox[i].ymin, (int)bbox[i].xmax, (int)bbox[i].ymax);
		printf("%f, %f, %f, %f\n", rcsq[i*4], rcsq[i*4+1], rcsq[i*4+2], rcsq[i*4+3]);
	}
	return 1;
}
