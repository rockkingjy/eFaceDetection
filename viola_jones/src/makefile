
CC = gcc
CXX = g++
OPENCV = -L/local/lib/ -lopencv_highgui -lopencv_core -lopencv_imgproc -I/local/include/opencv
CXXFLAGS += $(OPENCV) -std=c++11
LDFLAGS= `pkg-config --libs opencv` -lstdc++ -lm
HEADERS = $(wildcard *.h) $(wildcard *.hpp) $(wildcard */*.h) $(wildcard */*.hpp)
OBJS = 	AdaBoost.o  \
		ViolaJones.o \
		classifiers/CascadeClassifier.o \
		classifiers/Stage.o \
		classifiers/StrongClassifier.o \
		classifiers/WeakClassifier.o \
		features/Data.o \
		features/HaarFeatures.o \
		utils/Face.o \
		utils/IntegralImage.o \
		FaceDetector.o

all : vjface.bin
vjface.bin: $(OBJS) main.o 
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cc $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

clean:
	rm -rf */*/*.o */*.o *.o */*.bin *.bin *.so */*.so