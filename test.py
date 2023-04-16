import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO
import numpy
from numpy import cos,sin

path = 'C:/Users/ADMIN/Desktop/document/datasets/Data/data/'

def get_frames(path):
    cap = cv2.VideoCapture(path)
    images = []
    while True:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if frame is not None:
            images.append(frame)
            continue
        break
    return images

def show(images,frame):
    plt.imshow(images[frame])
    plt.show()

path = 'C:/Users/ADMIN/Desktop/document/datasets/yolodata/images/test.jpg'
image = cv2.imread(path)
