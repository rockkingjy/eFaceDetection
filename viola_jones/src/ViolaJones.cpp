/*

 */

#include "ViolaJones.h"

ViolaJones::ViolaJones() : AdaBoost()
{
	this->maxStages = 0;
	this->negativesPerLayer = 0;
	this->detectionWindowSize = 24;
	this->validationPath = "";
	this->numPositives = 0;
	this->numNegatives = 0;
	this->numValidation = 0;
	this->useNormalization = false;
}

ViolaJones::ViolaJones(string trainedPath) : AdaBoost()
{
	this->maxStages = 0;
	this->iterations = 0;
	this->detectionWindowSize = 24;
	this->classifier = *(new CascadeClassifier());
	this->negativesPerLayer = 0;
	this->validationPath = "";
	this->numPositives = 0;
	this->numNegatives = 0;
	this->numValidation = 0;
	this->useNormalization = false;
	loadTrainedData(trainedPath);
}

ViolaJones::ViolaJones(string positivePath, string negativePath, int maxStages, int numPositives, int numNegatives, int detectionWindowSize, int negativesPerLayer) : AdaBoost()
{
	this->iterations = 0;
	this->maxStages = maxStages;
	this->classifier = *(new CascadeClassifier());
	this->positivePath = positivePath;
	this->negativePath = negativePath;
	this->validationPath = "";
	this->useNormalization = false;
	this->detectionWindowSize = detectionWindowSize;
	this->features = {};
	this->numPositives = numPositives;
	this->numNegatives = numNegatives;
	this->numValidation = 0;
	if (negativesPerLayer == 0)
	{
		this->negativesPerLayer = numPositives;
	}
	else
	{
		this->negativesPerLayer = negativesPerLayer;
	}
}

/**
 * Prediction function based on cascade classifier
 */
int ViolaJones::predict(Mat img)
{
	return classifier.predict(img);
}

/**
 * Training Viola&Jones cascade classifiers. The cascade design process is driven from a set of detection
 * and performance goals. For the face detection task, past systems have achieved good detection rates
 * (between 85 and 95 percent) and extremely low false positive rates (on the order of 10−5 or 10−6).
 * The number of cascade stages and the size of each stage must be sufficient to achieve similar
 * detection performance while minimizing computation.
 */
void ViolaJones::train()
{
	cout << "Training ViolaJones face detector\n"
		 << endl;
	extractFeatures();

	float f = 0.5;
	float d = 0.98;
	float Ftarget = 0.00001;
	vector<int> featuresLayer{2, 5, 5, 5, 5, 5, 5, 5, 5, 5};// iterations for each stage
	float FPR = 1.;
	float DR = 0;
	float oldFPR = 1.;
	vector<WeakClassifier *> classifiers;

	int i = 0;
	int n;

	bool useValidation = false;
	if (validation.size() > 0)
	{
		useValidation = true;
	}

	while (FPR > Ftarget && i < maxStages)
	{
		if (negatives.size() == 0)
		{
			cout << "All training negative samples classified correctly. "
					"Could not achieve validation target FPR for this stage."
				 << endl;
			break;
		}

		n = 0;

		classifiers.clear();
		initializeWeights();

		//Rearrange features
		features.clear();
		features.reserve(positives.size() + negatives.size());
		features.insert(features.end(), positives.begin(), positives.end());
		features.insert(features.end(), negatives.begin(), negatives.end());

		if (!useValidation)
		{
			validation.clear();
			validation.reserve(negatives.size());
			validation.insert(validation.end(), negatives.begin(), negatives.end());
		}

		cout << "\n*** Stage n. " << i + 1 << " ***\n"
			 << endl;
		cout << "  -Training size: " << features.size() << endl;

		Stage *stage = new Stage(i + 1);
		classifier.addStage(stage);

		if (i < featuresLayer.size())
		{
			this->iterations = featuresLayer[i];
			cout << "  -Fixed number of classifiers: " << this->iterations << endl;
			StrongClassifier *strongClassifier = AdaBoost::train();
			stage->setClassifiers(strongClassifier->getClassifiers());
			stage->optimizeThreshold(positives, d);
			DR = evaluateDR(positives);
			stage->setDetectionRate(DR);
			FPR = evaluateFPR(validation);
			stage->setFpr(FPR);
			oldFPR = FPR;
		}
		else
		{
			cout << "  -Target FPR: " << (f * oldFPR) << endl;
			cout << "  -Target DR: " << (d) << "\n"
				 << endl;
			while (FPR > f * oldFPR || DR < d)
			{
				n++;
				this->iterations = n;

				//Train the current classifier
				StrongClassifier *strongClassifier = AdaBoost::train(classifiers);
				if (strongClassifier->getClassifiers().size() == 0)
				{
					cout << "Error training weak classifiers" << endl;
					return;
				}
				stage->setClassifiers(strongClassifier->getClassifiers());
				classifiers = strongClassifier->getClassifiers();

				//Optimizing stage threshold
				stage->optimizeThreshold(positives, d);
				//Evaluate current cascaded classifier on validation set to determine fpr & dr
				DR = evaluateDR(positives);
				stage->setDetectionRate(DR);
				FPR = evaluateFPR(validation);
				stage->setFpr(FPR);
			}
			oldFPR = FPR;
		}

		//Clear negative set
		negatives.clear();

		if (FPR > Ftarget)
		{
			//evaluate the current cascaded detector on the set of non-face images
			//and put any false detections into the set N.
			generateNegativeSet(negativesPerLayer, true);
		}
		stage->printInfo();
		i++;
		store();
	}
}

/**
 * Extract examples feature given images path. Generating positive, negative and validation set
 * for evaluating performance during training
 */
void ViolaJones::extractFeatures()
{
	//Loading training positive images
	int count = 0;
	Mat img, intImg;

	cout << "Extracting image features" << endl;
	auto t_start = chrono::high_resolution_clock::now();

	//Reading examples from folder
	vector<string> positiveImages = Utils::open(positivePath);
	vector<string> negativeImages = Utils::open(negativePath);
	//Shuffle negative examples (for variance)
	random_shuffle(positiveImages.begin(), positiveImages.end());
	random_shuffle(negativeImages.begin(), negativeImages.end());

	//Counting examples
	int totalExamples = numPositives + numNegatives;
	cout << "Training size: " << totalExamples << endl;
	if (numPositives > positiveImages.size())
		numPositives = positiveImages.size();
	cout << "  -Positive samples: " << numPositives << endl;
	cout << "  -Negative samples: " << numNegatives << endl;

	//Setting validation set (if defined)
	if (numValidation > 0)
	{
		vector<string> validationImages = Utils::open(validationPath);
		random_shuffle(validationImages.begin(), validationImages.end());
		totalExamples += numValidation;
		cout << "  -Validation set size: " << numValidation << endl;
		for (int k = 0; k < numValidation; ++k)
		{
			img = imread(validationPath + validationImages[k], CV_LOAD_IMAGE_GRAYSCALE);
			if (img.rows != 0 && img.cols != 0)
			{
				Mat dest;
				resize(img, dest, Size(detectionWindowSize, detectionWindowSize));
				if (useNormalization)
					normalizeImage(dest);
				intImg = IntegralImage::computeIntegralImage(dest);
				vector<float> features = HaarFeatures::extractFeatures(intImg, detectionWindowSize);
				validation.push_back(new Data(features, 0));
				count++;
				cout << "\rEvaluated: " << count + 1 << "/" << totalExamples << " images" << flush;
			}
		}
	}

	//Generating positive set
	for (int k = 0; k < numPositives; ++k)
	{
		img = imread(positivePath + positiveImages[k], CV_LOAD_IMAGE_GRAYSCALE);
		if (img.rows != 0 && img.cols != 0)
		{
			Mat dest;
			resize(img, dest, Size(detectionWindowSize, detectionWindowSize));
			if (useNormalization)
				normalizeImage(dest);
			intImg = IntegralImage::computeIntegralImage(dest);
			vector<float> features = HaarFeatures::extractFeatures(intImg, detectionWindowSize);
			positives.push_back(new Data(features, 1));
			count++;
			cout << "\rEvaluated: " << count + 1 << "/" << totalExamples << " images" << flush;
		}
	}

	//Generating negative set
	generateNegativeSet(numNegatives, false);

	cout << "\nExtracted features in ";
	auto t_end = chrono::high_resolution_clock::now();
	cout << std::fixed
		 << (chrono::duration<double, milli>(t_end - t_start).count()) / 1000
		 << " s\n"
		 << endl;
}


/**
 * Generating negative set for the next stage: iterates negative examples folder
 * looking for false positive examples and add them to the negative set.
 * Negative examples are shuffled for preventing overfitting.
 */
void ViolaJones::generateNegativeSet(int number, bool rotate)
{
	WeakClassifier *wc;
	for (int i = 0; i < classifier.getStages().size(); ++i)
	{
		for (int j = 0; j < classifier.getStages()[i]->getClassifiers().size(); ++j)
		{
			wc = classifier.getStages()[i]->getClassifiers()[j];
			HaarFeatures::getFeature(detectionWindowSize, wc);
		}
	}

	cout << "\nGenerating negative set for layer: max " << number << endl;
	vector<string> negativeImages = Utils::open(negativePath);
	random_shuffle(negativeImages.begin(), negativeImages.end());
	int count = 0;
	int evaluated = 0;
	int delta = 2;
	int maxRotate = -1;
	for (int k = 0; k < negativeImages.size() && count < number; ++k)
	{
		Mat img = imread(negativePath + negativeImages[k], CV_LOAD_IMAGE_GRAYSCALE);
		Mat dest, window;
		if (img.rows > 0 && img.cols > 0)
		{
			for (int j = 0; j < img.rows - detectionWindowSize - delta && count < number; j += delta)
			{
				for (int i = 0; i < img.cols - detectionWindowSize - delta && count < number; i += delta)
				{
					window = img(Rect(i, j, detectionWindowSize, detectionWindowSize));
					if (rotate)
					{
						maxRotate = 2;
					}
					for (int f = -2; f < maxRotate; ++f)
					{
						if (f > -2)
						{
							flip(window, dest, f);
						}
						else
						{
							dest = window;
						}
						evaluated++;
						if (useNormalization)
							normalizeImage(dest);
						Mat intImg = IntegralImage::computeIntegralImage(dest);
						if (classifier.predict(intImg) == 1)
						{
							vector<float> features = HaarFeatures::extractFeatures(intImg, detectionWindowSize);
							negatives.push_back(new Data(features, 0));
							count++;
						}
						cout << "\rAdded " << count << " (" << evaluated << " tested) images to the negative set" << flush;
					}
				}
			}
		}
	}
}

/**
 * Given an input set of detections (Faces with a rectangle and a score), merge detections
 * in the way explained in Viola&Jones article.
 * The set of detections are first partitioned into disjoint subsets. Two detections are
 * in the same subset if their bounding regions overlap. Each partition yields a single
 * final detection. The corners of the final bounding region are the average of the
 * corners of all detections in the set.
 */
vector<Face> ViolaJones::mergeDetections(vector<Face> &detections, int padding, float th)
{
	vector<Face> output, cluster;
	float score;
	Rect a, b;

	sort(detections.begin(), detections.end(),
		 [](Face const &a, Face const &b) { return a.getRect().area() > b.getRect().area(); });

	for (unsigned int i = 0; i < detections.size(); ++i)
	{
		if (!detections[i].isEvaluated())
		{
			cluster.clear();
			cluster.push_back(detections[i]);
			detections[i].setEvaluated(true);
			a = detections[i].getRect();

			for (unsigned int j = 0; j < detections.size(); ++j)
			{
				if (i != j && !detections[j].isEvaluated())
				{
					b = detections[j].getRect();
					score = (float)(a & b).area() / (a | b).area();
					if (score > th)
					{
						detections[j].setEvaluated(true);
						cluster.push_back(detections[j]);
					}
				}
			}

			if (cluster.size() > 3)
			{
				Rect result(0, 0, 0, 0);

				for (unsigned int k = 0; k < cluster.size(); ++k)
				{
					result.x += cluster[k].getRect().x;
					result.y += cluster[k].getRect().y;
					result.width += cluster[k].getRect().width;
					result.height += cluster[k].getRect().height;
				}

				result.x = result.x / cluster.size() - padding;
				result.y = result.y / cluster.size() - padding;
				result.width = result.width / cluster.size() + 2 * padding;
				result.height = result.height / cluster.size() + 2 * padding;

				output.push_back(Face(result, (float)cluster.size()));
			}
		}
	}
	return output;
}

void ViolaJones::normalizeImage(Mat &img)
{
	int width = img.cols;
	int height = img.rows;
	int N = width * height;
	Mat intImg, intImgSq;

	//Compute mean and std value
	intImg = IntegralImage::computeIntegralImage(img);
	float mean = (float)intImg.at<float>(height - 1, width - 1) / N;
	intImgSq = IntegralImage::computeIntegralSquaredImage(img, mean);
	float stdev = sqrt((float)intImgSq.at<float>(height - 1, width - 1) / N);
	if (stdev == 0)
		stdev = 1;

	Scalar intensity;
	float value;
	Mat tmp(height, width, CV_32F);
	for (int r = 0; r < img.rows; ++r)
	{
		for (int c = 0; c < img.cols; ++c)
		{
			intensity = img.at<uchar>(r, c);
			value = (((float)intensity[0]) - mean) / stdev;
			tmp.at<float>(r, c) = value;
		}
	}

	double min, max;
	minMaxLoc(tmp, &min, &max);

	for (int r = 0; r < tmp.rows; ++r)
	{
		for (int c = 0; c < tmp.cols; ++c)
		{
			img.at<uchar>(r, c) = (int)(255 / (max - min) * (tmp.at<float>(r, c) - min));
		}
	}
}

/**
 * Evaluating False Positive Rate on the validation set
 */
float ViolaJones::evaluateFPR(vector<Data *> &validationSet)
{
	cout << "Evaluate FPR on validation set:" << endl;
	int fp = 0;
	int tn = 0;
	int prediction;
	for (int i = 0; i < validationSet.size(); ++i)
	{
		prediction = classifier.predict(validationSet[i]->getFeatures());
		if (prediction == 1 && validationSet[i]->getLabel() == 0)
		{
			fp++;
		}
		else if (prediction == 0 && validationSet[i]->getLabel() == 0)
		{
			tn++;
		}
	}
	float fpr = (float)fp / (fp + tn);
	cout << "FPR: " << fpr;
	cout << " (FP: " << fp << " TN: " << tn << ")\n"
		 << endl;
	return fpr;
}

/**
 * Evaluating Detection Rate on the validation set
 */
float ViolaJones::evaluateDR(vector<Data *> &validationSet)
{
	int tp = 0;
	int fn = 0;
	int prediction;
	for (int i = 0; i < validationSet.size(); ++i)
	{
		prediction = classifier.predict(validationSet[i]->getFeatures());
		if (prediction == 0 && validationSet[i]->getLabel() == 1)
		{
			fn++;
		}
		else if (prediction == 1 && validationSet[i]->getLabel() == 1)
		{
			tp++;
		}
	}
	float dr = (float)tp / (tp + fn);
	cout << "DR: " << dr;
	cout << " (TP: " << tp << " FN: " << fn << ")" << endl;
	return dr;
}

/**
 * Overwrite AdaBoost functions
 */
float ViolaJones::updateAlpha(float error)
{
	if (error < 0.0001)
	{
		return 1000;
	}
	return log((1 - error) / error);
}

float ViolaJones::updateBeta(float error)
{
	return error / (1 - error);
}

void ViolaJones::updateWeights(WeakClassifier *weakClassifier)
{
	int e, prediction;
	float num;
	for (int i = 0; i < features.size(); ++i)
	{
		e = (weakClassifier->predict(this->features[i]) == features[i]->getLabel()) ? 0 : 1;
		num = features[i]->getWeight() * (pow(weakClassifier->getBeta(), (float)(1 - e)));
		features[i]->setWeight(num);
	}
}

void ViolaJones::initializeWeights()
{
	for (int i = 0; i < positives.size(); ++i)
	{
		positives[i]->setWeight((float)1 / (2 * positives.size()));
	}
	for (int i = 0; i < negatives.size(); ++i)
	{
		negatives[i]->setWeight((float)1 / (2 * negatives.size()));
	}
}

/**
 * Storing face detector in a textual form in order to reuse it in
 * the future without train it again.
 */
void ViolaJones::store()
{
	cout << "\nStoring trained face detector" << endl;
	ofstream output, data;
	output.open("trainedInfo.txt");
	data.open("trainedData.txt");

	WeakClassifier *wc;

	for (unsigned int i = 0; i < classifier.getStages().size(); ++i)
	{
		Stage *stage = classifier.getStages()[i];

		//Outputs info
		output << "Stage " << i << "\n\n";
		output << "FPR: " << stage->getFpr() << "\n";
		output << "DR: " << stage->getDetectionRate() << "\n";
		output << "Threshold: " << stage->getThreshold() << "\n";
		output << "Classifiers:\n"
			   << endl;
		//Output data
		data << "s:" << stage->getFpr() << "," << stage->getDetectionRate()
			 << "," << stage->getThreshold() << "\n";

		for (unsigned int j = 0; j < stage->getClassifiers().size(); ++j)
		{
			wc = stage->getClassifiers()[j];
			output << "WeakClassifier " << j << "\n";
			output << "Error: " << wc->getError() << "\n";
			output << "Dimension: " << wc->getDimension() << "\n";
			output << "Threshold: " << wc->getThreshold() << "\n";
			output << "Alpha: " << wc->getAlpha() << "\n";
			output << "Beta: " << wc->getBeta() << "\n";
			if (wc->getSign() == POSITIVE)
			{
				output << "Sign: POSITIVE\n";
			}
			else
			{
				output << "Sign: NEGATIVE\n";
			}
			output << "Misclassified: " << wc->getMisclassified() << "\n\n";
			//Outputs data
			data << "c:" << wc->getError() << "," << wc->getDimension() << ","
				 << wc->getThreshold() << "," << wc->getAlpha() << ","
				 << wc->getBeta() << ",";
			if (wc->getSign() == POSITIVE)
			{
				data << "POSITIVE,";
			}
			else
			{
				data << "NEGATIVE,";
			}
			data << wc->getMisclassified() << "\n";
		}

		output << "---------------\n"
			   << endl;
	}

	output.close();
}

/**
 * Loads cascade detector from given file (the file must be correctly formatted
 * as in the store function output
 */
void ViolaJones::loadTrainedData(string filename)
{
	cout << "Loading data from file: " << filename << endl;
	string line;
	string read;
	ifstream readFile(filename);
	Stage *stage;
	WeakClassifier *wc;

	while (getline(readFile, line))
	{
		stringstream iss(line);
		getline(iss, read, ':');
		if (read.compare("s") == 0)
		{
			//Found stage
			stage = new Stage(classifier.getStages().size());
			getline(iss, read, ',');
			stage->setFpr(stod(read));
			getline(iss, read, ',');
			stage->setDetectionRate(stod(read));
			getline(iss, read, ',');
			stage->setThreshold(stod(read));
			classifier.addStage(stage);
		}
		else if (read.compare("c") == 0)
		{
			//Found classifier
			wc = new WeakClassifier();
			getline(iss, read, ',');
			wc->setError(stod(read));
			getline(iss, read, ',');
			wc->setDimension(stoi(read));
			getline(iss, read, ',');
			wc->setThreshold(stod(read));
			getline(iss, read, ',');
			wc->setAlpha(stod(read));
			getline(iss, read, ',');
			wc->setBeta(stod(read));
			getline(iss, read, ',');
			if (read.compare("positive") == 0)
			{
				wc->setSign(POSITIVE);
			}
			else
			{
				wc->setSign(NEGATIVE);
			}
			getline(iss, read, ',');
			wc->setMisclassified(stoi(read));
			HaarFeatures::getFeature(detectionWindowSize, wc);
			stage->addClassifier(wc);
		}
	}

	readFile.close();
	cout << "Trained data loaded correctly" << endl;
}

// ---
const string &ViolaJones::getValidationPath() const
{
	return validationPath;
}

int ViolaJones::getMaxStages() const
{
	return maxStages;
}

void ViolaJones::setMaxStages(int maxStages)
{
	this->maxStages = maxStages;
}

const string &ViolaJones::getNegativePath() const
{
	return negativePath;
}

void ViolaJones::setNegativePath(const string &negativePath)
{
	this->negativePath = negativePath;
}

int ViolaJones::getNegativesPerLayer() const
{
	return negativesPerLayer;
}

void ViolaJones::setNegativesPerLayer(int negativesPerLayer)
{
	this->negativesPerLayer = negativesPerLayer;
}

int ViolaJones::getNumNegatives() const
{
	return numNegatives;
}

void ViolaJones::setNumNegatives(int numNegatives)
{
	this->numNegatives = numNegatives;
}

int ViolaJones::getNumPositives() const
{
	return numPositives;
}

void ViolaJones::setNumPositives(int numPositives)
{
	this->numPositives = numPositives;
}

const string &ViolaJones::getPositivePath() const
{
	return positivePath;
}

void ViolaJones::setPositivePath(const string &positivePath)
{
	this->positivePath = positivePath;
}

bool ViolaJones::isUseNormalization() const
{
	return useNormalization;
}

void ViolaJones::setUseNormalization(bool useNormalization)
{
	this->useNormalization = useNormalization;
}

void ViolaJones::setClassifier(const CascadeClassifier &classifier)
{
	this->classifier = classifier;
}

const CascadeClassifier &ViolaJones::getClassifier() const
{
	return classifier;
}

void ViolaJones::setValidationSet(const string &validationPath, int examples)
{
	this->validationPath = validationPath;
	if (examples == 0)
	{
		vector<string> validationImages = Utils::open(validationPath);
		this->numValidation = validationImages.size();
	}
	else
	{
		this->numValidation = examples;
	}
}
