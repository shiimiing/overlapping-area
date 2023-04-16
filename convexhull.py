from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

def preprocess(image):
    chull = convex_hull_image(image, tolerance=1)
    return chull.astype(np.float32)

def dist(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def corner(img):
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    corner = [[y,x] for x in range(dst.shape[0]) for y in range(dst.shape[1]) if x != y and dst[x][y] > 0 and x < 200]
    real_conner = [corner.pop(0)]
    for val in corner:
        check = False
        for rv in real_conner:
            if val in real_conner:
                continue
            if dist(val,rv) < 150:
                check = True
                break
        if not check:
            real_conner.append(val)
    minn = min(real_conner)
    real_conner.sort(key = lambda x: x[0])
    real_conner.pop(real_conner.index(minn))
    return real_conner

def getAllCorner(img):
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    corner = [[y,x] for x in range(dst.shape[0]) for y in range(dst.shape[1]) if x != y and dst[x][y] > 0 and x < 200]
    return corner
    