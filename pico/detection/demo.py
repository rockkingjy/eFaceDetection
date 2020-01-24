import subprocess
import numpy
import os
import ctypes
import cv2
import time

def process_frame(gray, cascade, slot, nmemslots, maxslotsize):
    maxndets = 2048
    dets = numpy.zeros(4*maxndets, dtype=numpy.float32)

    ndets = pico.find_objects(
        ctypes.c_void_p(dets.ctypes.data),
        ctypes.c_int(maxndets),
        ctypes.c_void_p(cascade.ctypes.data),
        ctypes.c_float(0.0),
        ctypes.c_void_p(gray.ctypes.data),
        ctypes.c_int(gray.shape[0]),
        ctypes.c_int(gray.shape[1]),
        ctypes.c_int(gray.shape[1]),
        ctypes.c_float(1.1),
        ctypes.c_float(0.1),
        ctypes.c_float(100),
        ctypes.c_float(1000)
    )

    ndets = pico.update_memory(
        ctypes.c_void_p(slot.ctypes.data),
        ctypes.c_void_p(memory.ctypes.data),
        ctypes.c_void_p(counts.ctypes.data),
        ctypes.c_int(nmemslots),
        ctypes.c_int(maxslotsize),
        ctypes.c_void_p(dets.ctypes.data),
        ctypes.c_int(ndets),
        ctypes.c_int(maxndets)
    )

    ndets = pico.cluster_detections(
        ctypes.c_void_p(dets.ctypes.data),
        ctypes.c_int(ndets)
    )

    return list(dets.reshape(-1, 4))[0:ndets]

if __name__ == "__main__":
	# compile and create .so file
	os.system('cc picornt.c -O3 -fPIC -shared -o picornt.lib.so')
	pico = ctypes.cdll.LoadLibrary('./picornt.lib.so')
	os.system('rm picornt.lib.so')

	# weight
	weight = 'face_15.weight'
	bytes = open(weight, 'rb').read()
	cascade = numpy.frombuffer(bytes, dtype=numpy.uint8)

	slot = numpy.zeros(1, dtype=numpy.int32)
	nmemslots = 5
	maxslotsize = 1024
	memory = numpy.zeros(4*nmemslots*maxslotsize, dtype=numpy.float32)
	counts = numpy.zeros(nmemslots, dtype=numpy.int32)

	cap = cv2.VideoCapture(0)
	while(True):
		ret, frm = cap.read()
		if max(frm.shape[0], frm.shape[1]) > 640:
			frm = cv2.resize(frm, (0, 0), fx=0.5, fy=0.5)
		gray = numpy.ascontiguousarray(
			frm[:, :, 1].reshape((frm.shape[0], frm.shape[1])))

		start = time.time()
		# gray needs to be numpy.uint8 array
		dets = process_frame(gray, cascade, slot, nmemslots, maxslotsize)  
		for det in dets:
			if det[3] >= 50.0:
				cv2.circle(frm, (int(det[1]), int(det[0])),
						int(det[2]/2.0), (0, 0, 255), 4)
		end = time.time()
		print("time:{} ms".format((end - start)*1000))

		#
		cv2.imshow('...', frm)
		if cv2.waitKey(5) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

