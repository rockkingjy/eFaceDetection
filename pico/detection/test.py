import subprocess
import numpy
import os
import ctypes
import cv2
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def process_frame(gray, pico, cascade, slot, memory, counts, nmemslots, maxslotsize):
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


def test_caltechfaces():
    # compile and create .so file
    os.system('cc picornt.c -O3 -fPIC -shared -o picornt.lib.so')
    pico = ctypes.cdll.LoadLibrary('./picornt.lib.so')
    os.system('rm picornt.lib.so')

    # load weight
    weight = '../weights/face_15.weight'
    bytes = open(weight, 'rb').read()
    cascade = numpy.frombuffer(bytes, dtype=numpy.uint8)

    #
    slot = numpy.zeros(1, dtype=numpy.int32)
    nmemslots = 5
    maxslotsize = 1024
    memory = numpy.zeros(4*nmemslots*maxslotsize, dtype=numpy.float32)
    counts = numpy.zeros(nmemslots, dtype=numpy.int32)

    # read file
    annots = open(os.path.join('../caltechfaces/',
                               'WebFaces_GroundThruth.txt'), 'r')
    imgnames = []
    faces = []
    dict = {}
    for line in annots.readlines():
        if line.strip() != '':
            imgname = line.split()[0]
            # print(imgname)
            if imgname in dict:
                i = dict[imgname]
                faces[i].append([float(x) for x in line.split()[1:]])
            else:
                dict[imgname] = len(imgnames)
                imgnames.append(imgname)
                faces.append([[float(x) for x in line.split()[1:]]])

    # create folder for save gt and det
    gt_path = '../weights/gt/'
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    det_path = '../weights/det/'
    if not os.path.exists(det_path):
        os.makedirs(det_path)

    for i in range(0, len(imgnames)):
        img = cv2.imread('../caltechfaces/' + imgnames[i])
        height, width, channels = img.shape
        bboxes = []
        for face in faces[i]:
            eyedist = ((face[0]-face[2])**2 + (face[1]-face[3])**2)**0.5
            r = (face[1]+face[3])/2.0 + 0.25*eyedist
            c = (face[0]+face[2])/2.0
            s = 2.0*1.5*eyedist
            xl = max(int(c - s), 0)
            yl = max(int(r - s), 0)
            xr = min(int(c + s), width)
            yr = min(int(c + s), height)
            bboxes.append((xl, yl, xr, yr))

        # write ground truth file
        name = str(gt_path) + str(os.path.splitext(imgnames[i])[0]) + '.txt'
        print(name)
        f = open(name, 'w')
        print(bboxes)
        for box in bboxes:
            f.write("face " + str(box[0]) + " " + str(box[1]
                                                      ) + " " + str(box[2]) + " " + str(box[3]))
            f.write("\n")
        f.close()

        # write detection
        name = str(det_path) + str(os.path.splitext(imgnames[i])[0]) + '.txt'
        print(name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if max(img_gray.shape[0], img_gray.shape[1]) > 640:
            img_gray = cv2.resize(img_gray, (0, 0), fx=0.5, fy=0.5)
        gray = numpy.ascontiguousarray(
            img_gray[:, :].reshape((img_gray.shape[0], img_gray.shape[1])))

        dets = process_frame(gray, pico, cascade, slot,
                             memory, counts, nmemslots, maxslotsize)
        bboxes = []
        for det in dets:
            if det[3] >= 50.0:
                xl = max(int(det[1] - det[2]/2.0), 0)
                yl = max(int(det[0] - det[2]/2.0), 0)
                xr = min(int(det[1] + det[2]/2.0), width)
                yr = min(int(det[0] + det[2]/2.0), height)
                bboxes.append((xl, yl, xr, yr))
                #cv2.circle(img_gray, (int(det[1]), int(det[0])), int(det[2]/2.0), (0, 0, 255), 4)

        # save detection result
        f = open(name, 'w')
        print(bboxes)
        for box in bboxes:
            f.write("face 1 " + str(box[0]) + " " + str(box[1]
                                                        ) + " " + str(box[2]) + " " + str(box[3]))
            f.write("\n")
        f.close()
        """
        cv2.imshow('...', img_gray)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            continue
        #exit(0)
        """

        """
        # draws    
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        plt.imshow(img)
        rec = Rectangle((int(xl), int(yl)), int(2*s), int(2*s))
        #rec = Circle((c, r), s)
        ax.add_patch(rec)
        plt.show()
        """


def run_one_image():
    imgpath = "../imgs/ex_face_2.jpg"

    # compile and create .so file
    os.system('cc picornt.c -O3 -fPIC -shared -o picornt.lib.so')
    pico = ctypes.cdll.LoadLibrary('./picornt.lib.so')
    os.system('rm picornt.lib.so')

    # weight
    weight = '../weights/face_15.weight'
    bytes = open(weight, 'rb').read()
    cascade = numpy.frombuffer(bytes, dtype=numpy.uint8)

    slot = numpy.zeros(1, dtype=numpy.int32)
    nmemslots = 5
    maxslotsize = 1024
    memory = numpy.zeros(4*nmemslots*maxslotsize, dtype=numpy.float32)
    counts = numpy.zeros(nmemslots, dtype=numpy.int32)

    frm = cv2.imread(imgpath)
    if max(frm.shape[0], frm.shape[1]) > 640:
        frm = cv2.resize(frm, (0, 0), fx=0.5, fy=0.5)
    gray = numpy.ascontiguousarray(
        frm[:, :, 1].reshape((frm.shape[0], frm.shape[1])))

    start = time.time()
    # gray needs to be numpy.uint8 array
    dets = process_frame(gray, pico, cascade, slot, memory,
                         counts, nmemslots, maxslotsize)
    for det in dets:
        if det[3] >= 50.0:
            cv2.circle(frm, (int(det[1]), int(det[0])),
                       int(det[2]/2.0), (0, 0, 255), 4)
    end = time.time()
    print("time:{} ms".format((end - start)*1000))

    #
    cv2.imshow('...', frm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera():
        # compile and create .so file
    os.system('cc picornt.c -O3 -fPIC -shared -o picornt.lib.so')
    pico = ctypes.cdll.LoadLibrary('./picornt.lib.so')
    os.system('rm picornt.lib.so')

    # weight
    weight = '../weights/face_15.weight'
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
        dets = process_frame(gray, pico, cascade, slot,
                             memory, counts, nmemslots, maxslotsize)
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


if __name__ == "__main__":

    # test_caltechfaces()
    run_camera()
    #run_one_image()
