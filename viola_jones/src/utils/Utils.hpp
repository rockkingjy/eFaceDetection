/*

 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dirent.h>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

enum example
{
	POSITIVE,
	NEGATIVE
};

//Sign function
template <typename T>
int sgn(T val)
{
	return (T(0) < val) - (val < T(0));
}

class Utils
{
public:
	static vector<string> open(string path = ".")
	{
		DIR *dir;
		dirent *pdir;
		vector<string> files;

		dir = opendir(path.c_str());

		while (pdir = readdir(dir))
		{
			if (strcmp(pdir->d_name, ".") != 0 && strcmp(pdir->d_name, "..") != 0)
				files.push_back(pdir->d_name);
		}

		closedir(dir);
		return files;
	}

	static void generateNonFacesDataset(string path, string outputDir, int number, int size)
	{
		cout << "Generating non faces dataset from given images" << endl;
		vector<string> images = open(path);
		int counter = 0;
		int k = 0;
		int delta = 5;
		stringstream ss;
		Mat window;
		while (k < images.size() && counter < number)
		{
			Mat img = imread(path + "/" + images[k], CV_LOAD_IMAGE_GRAYSCALE);
			if (img.cols != 0 && img.rows != 0)
			{
				resize(img, img, Size(200, 100));
				for (int j = 0; j < img.rows - size - delta && counter < number; j += delta)
				{
					for (int i = 0; i < img.cols - size - delta && counter < number; i += delta)
					{
						window = img(Rect(i, j, size, size));
						ss.str("");
						ss << outputDir << "/image_val_" << counter << ".pgm";
						imwrite(ss.str(), window);
						counter++;
						cout << "\rGenerated: " << counter << "/" << number << " images" << flush;
					}
				}
			}
			k++;
		}
	}

	static Mat rotate(Mat src, float angle)
	{
		Mat dst;
		Point2f pt(src.cols / 2., src.rows / 2.);
		Mat r = getRotationMatrix2D(pt, angle, 1.0);
		warpAffine(src, dst, r, Size(src.cols, src.rows));
		return dst;
	}
};

#endif /* UTILS_HPP_ */
