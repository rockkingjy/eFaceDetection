import csv
import cv2
import sys
import os
import struct
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# h, w, im[h*w]
def write_rid_to_stdout(im, f_out):
    h = im.shape[0]
    w = im.shape[1]
    hw = struct.pack('ii', h, w)
    pixels = struct.pack('%sB' % h*w, *im.reshape(-1))
    f_out.write(hw)
    f_out.write(pixels)

# h, w, im[h*w], len(bboxes), box[3]xlen(bboxes)
def write_sample_to_stdout(img, bboxes, f_out):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    write_rid_to_stdout(img, f_out)
    f_out.write(struct.pack('i', len(bboxes)))
    for box in bboxes:
        f_out.write(struct.pack('iii', box[0], box[1], box[2]))


def visualize_bboxes(img, bboxes, f_out):
    for box in bboxes:
        cv2.circle(img, (int(box[1]), int(box[0])), int(
            box[2]/2.0), (0, 0, 255), thickness=2)
    cv2.imshow('...', img)
    cv2.waitKey(0)

# change scale, flip to get more positive samples
def export_img_and_boxes(img, bboxes, f_out):
    for i in range(0, 8):  # 7
        # resize
        scalefactor = 0.7 + 0.6*numpy.random.random()
        resized_img = cv2.resize(img, (0, 0), fx=scalefactor, fy=scalefactor)
        # flip
        flip = numpy.random.random() < 0.5
        if flip:
            resized_img = cv2.flip(resized_img, 1)

        resized_bboxes = []
        for box in bboxes:
            if flip:
                resized_box = (int(scalefactor*box[0]), resized_img.shape[1] - int(
                    scalefactor*box[1]), int(scalefactor*box[2]))
            else:
                resized_box = (
                    int(scalefactor*box[0]), int(scalefactor*box[1]), int(scalefactor*box[2]))
            if resized_box[2] >= 24:
                resized_bboxes.append(resized_box)

        write_sample_to_stdout(resized_img, resized_bboxes, f_out)
        #visualize_bboxes(resized_img, resized_bboxes)


def read_face(root, output_name):
    with open(root + 'loose_landmark_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(row)
                line_count += 1
            else:
                print(row)
                line_count += 1
                if line_count > 1000:
                    im_path = root + 'test/' + row[0] + '.jpg'
                    img = cv2.imread(im_path)
                    x = []
                    y = []
                    for i in range(5):
                        x.append(float(row[i*2+1]))
                        y.append(float(row[i*2+2]))
                    #print(x)
                    #print(y)
                    """
                    fig, ax = plt.subplots(1)
                    ax.set_aspect('equal')
                    plt.imshow(img)
                    
                    # draw small circle for the landmarks
                    # l/r eye, nose, l/r mouth corners
                    i = 1
                    for xx, yy in zip(x, y):
                        circ = Circle((xx, yy), i)
                        i = i+1
                        ax.add_patch(circ)
                    
                    
                    # calculate the area of mouth
                    xl = min(x[2], x[3], x[4])
                    w = max(x[2], x[3], x[4]) - xl
                    yl = y[2]
                    #yl = min((y[2] + y[3])/2, (y[2] + y[4])/2)
                    h = min(y[3] - yl, y[4] - yl) * 2 + abs(y[3] - y[4])
                    rec = Rectangle((xl, yl), w, h)
                    

                    r = int((y[3] + y[4]) / 2)
                    c = int((x[3] + x[4]) / 2)
                    dis_l = ((r - y[3])**2 + (c - x[3])**2)**0.5
                    dis_r = ((r - y[4])**2 + (c - x[4])**2)**0.5
                    s = int(max(dis_l, dis_r))
                    bboxes = []
                    bboxes.append((r, c, s))
                    """

                    eyedist = ((x[0]-x[1])**2 + (y[0]-y[1])**2)**0.5
                    r = int((y[0]+y[1])/2.0 + 0.25*eyedist)
                    c = int((x[0]+x[1])/2.0)
                    s = int(2.0*1.5*eyedist)
                    bboxes = []
                    bboxes.append((r, c, s))
                    """
                    # draw the mouth patch
                    rec = Circle((c, r), s)
                    ax.add_patch(rec)
                    plt.show()
                    #exit(0)
                    """
                    
                    f_out = open(output_name, mode='ab')
                    write_sample_to_stdout(img, bboxes, f_out)
                    f_out.close()
                    #export_img_and_boxes(img, bboxes, f_out)
                    

if __name__ == '__main__':
    root = '/home/neusoft/data/mouth/'
    output_name = 'data_face_big_15.dat'
    read_face(root, output_name)
