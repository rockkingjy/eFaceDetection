
OPENCV = -lopencv_highgui -lopencv_core -lopencv_imgproc #-I$(HOME)/anaconda3/include -L$(HOME)/anaconda3/lib

output:
	g++ sample.c ./picornt.c -lrt $(OPENCV) -o pico -O3

clean:
	rm pico