from datetime import datetime
from PIL import Image
import numpy as np
import struct
import random
import base64
import cv2

def read_bin(featFn):
    with open(featFn, "rb") as f:
        featLen, _, _, _ = struct.unpack("4i", f.read(16))
        return np.array(struct.unpack("%df" % (featLen), f.read()))

def write_bin(feats, featFn):
    feature = list(feats)
    with open(featFn, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))

def draw_box_name(bbox,name,frame):
    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
    frame = cv2.putText(frame,
                    name,
                    (bbox[0],bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0,255,0),
                    3,
                    cv2.LINE_AA)
    return frame

def findBigFace(bounding_boxes):
    idx = -1
    maxSize = 0
    for i in range(len(bounding_boxes)):
        width = int(bounding_boxes[i][2]) - int(bounding_boxes[i][0])
        height = int(bounding_boxes[i][3]) - int(bounding_boxes[i][1])
        size =  width * height
        if size > maxSize:
            maxSize = size
            idx = i
    return idx