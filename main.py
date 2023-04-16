from convexhull import *
import cv2
import numpy
from ultralytics import YOLO
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import tkinter
import os
import time
matplotlib.use('TkAgg')

def get3point(pp):
    return pp[:3]

src_img = [320,192]

def cam_c1(img,pts1):
    pts1 = numpy.float32(pts1)
    pts2 = numpy.float32([[src_img[0]/5,src_img[1]],[0,src_img[1]],[src_img[0],src_img[1]/5]])
    M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(img,M,(900,700))

def cam_c2(img,pts1):
    pts1 = numpy.float32(pts1)
    pts2 = numpy.float32([[src_img[0],src_img[1]/5],[src_img[0],src_img[1]],[src_img[0]/5,src_img[1]]])
    M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(img,M,(900,700))

def cam_c3(img,pts1):
    pts1 = numpy.float32(pts1)
    pts2 = numpy.float32([[0,src_img[1]/5],[0,0],[src_img[0]/5,0]])
    M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(img,M,(900,700))

def cam_c4(img,pts1):
    pts1 = numpy.float32(pts1)
    pts2 = numpy.float32([[src_img[0]/5,0],[src_img[0],0],[src_img[0],src_img[1]/5]])
    M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(img,M,(900,700))

def rv_cam_c1(img,pts1):
    pts1 = numpy.float32(pts1)
    pts2 = numpy.float32([[src_img[0]/5,src_img[1]],[0,src_img[1]],[src_img[0],src_img[1]/5]])
    M = cv2.getAffineTransform(pts2,pts1)
    return cv2.warpAffine(img,M,(640, 384))

def rv_cam_c2(img,pts1):
    pts1 = numpy.float32(pts1)
    pts2 = numpy.float32([[src_img[0],src_img[1]/5],[src_img[0],src_img[1]],[src_img[0]/5,src_img[1]]])
    M = cv2.getAffineTransform(pts2,pts1)
    return cv2.warpAffine(img,M,(640, 384))

def rv_cam_c3(img,pts1):
    pts1 = numpy.float32(pts1)
    pts2 = numpy.float32([[0,src_img[1]/5],[0,0],[src_img[0]/5,0]])
    M = cv2.getAffineTransform(pts2,pts1)
    return cv2.warpAffine(img,M,(640, 384))

def rv_cam_c4(img,pts1):
    pts1 = numpy.float32(pts1)
    pts2 = numpy.float32([[src_img[0]/5,0],[src_img[0],0],[src_img[0],src_img[1]/5]])
    M = cv2.getAffineTransform(pts2,pts1)
    return cv2.warpAffine(img,M,(640, 384))
#fdffg
model = YOLO('best (1).pt')
def extractOutput(path):

    video_cam1 = cv2.VideoCapture(path+'/'+'CAM_1.mp4')
    video_cam2 = cv2.VideoCapture(path+'/'+'CAM_2.mp4')
    video_cam3 = cv2.VideoCapture(path+'/'+'CAM_3.mp4')
    video_cam4 = cv2.VideoCapture(path+'/'+'CAM_4.mp4')

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    count = 1
    while True:
        a=time.time()
        res1, frame_cam1 = video_cam1.read()
        res2, frame_cam2 = video_cam2.read()
        res3, frame_cam3 = video_cam3.read()
        res4, frame_cam4 = video_cam4.read()

        raw1 = frame_cam1
        raw2 = frame_cam2
        raw3 = frame_cam3
        raw4 = frame_cam4
        if not res1: 
            break
        if not res2: 
            break
        if not res3: 
            break
        if not res4: 
            break

        if model(frame_cam1)[0].masks is None:
            continue
        if model(frame_cam2)[0].masks is None:
            continue
        if model(frame_cam3)[0].masks is None:
            continue
        if model(frame_cam4)[0].masks is None:
            continue

        total1 = model(frame_cam1)[0].masks.data[0]
        total2 = model(frame_cam2)[0].masks.data[0]
        total3 = model(frame_cam3)[0].masks.data[0]
        total4 = model(frame_cam4)[0].masks.data[0]

        total1 = total1.clone().detach().cpu()
        total2 = total2.clone().detach().cpu()
        total3 = total3.clone().detach().cpu()
        total4 = total4.clone().detach().cpu()

        total1 = numpy.array(total1)
        total2 = numpy.array(total2)
        total3 = numpy.array(total3)
        total4 = numpy.array(total4)

        total1 = preprocess(total1)
        total2 = preprocess(total2)
        total3 = preprocess(total3)
        total4 = preprocess(total4)

        cors1 = list(get3point(corner(total1)))
        cors2 = list(get3point(corner(total2)))
        cors3 = list(get3point(corner(total3)))
        cors4 = list(get3point(corner(total4)))

        mapmat1 = cv2.resize(frame_cam1, (640, 384))
        mapmat2 = cv2.resize(frame_cam2, (640, 384))
        mapmat3 = cv2.resize(frame_cam3, (640, 384))
        mapmat4 = cv2.resize(frame_cam4, (640, 384))
        
        cors1.append([0,0])
        cors2.append([0,0])
        cors3.append([0,0])
        cors4.append([0,0])

        image1 = cv2.bitwise_and(total1,cv2.rotate(total4, cv2.ROTATE_180), mask = None)
        image1 = preprocess(image1)

        image2 = cv2.bitwise_and(total2,cv2.rotate(total3, cv2.ROTATE_180), mask = None)
        image2 = preprocess(image2)

        image3 = cv2.bitwise_and(total3,cv2.rotate(total2, cv2.ROTATE_180), mask = None)
        image3 = preprocess(image3)

        image4 = cv2.bitwise_and(total4,cv2.rotate(total1, cv2.ROTATE_180), mask = None)
        image4 = preprocess(image4)
        b=time.time()
        corner1 = getAllCorner(image1)
        temp1 = [f"{i[0]},{i[1]}" for i in corner1]
        raw1 = ",".join(temp1)
        raw1 = f"({raw1})"
        f = open(path+'/CAM_1.txt', 'a')
        f.write(f"frame_{count}.jpg, {raw1}, {b-a}/n")
        f.close()
        b=time.time()
        corner2 = getAllCorner(image2)
        temp2 = [f"{i[0]},{i[1]}" for i in corner2]
        raw2 = ",".join(temp2)
        raw2 = f"({raw2})"
        f = open(path+'/CAM_2.txt', 'a')
        f.write(f"frame_{count}.jpg, {raw2}, {b-a}/n")
        f.close()
        b=time.time()
        corner3 = getAllCorner(image3)
        temp3 = [f"{i[0]},{i[1]}" for i in corner3]
        raw3 = ",".join(temp3)
        raw3 = f"({raw3})"
        f = open(path+'/CAM_3.txt', 'a')
        f.write(f"frame_{count}.jpg, {raw3}, {b-a}/n")
        f.close()
        b=time.time()
        corner4 = getAllCorner(image4)
        temp4 = [f"{i[0]},{i[1]}" for i in corner4]
        raw4 = ",".join(temp4)
        raw4 = f"({raw4})"
        f = open(path+'/CAM_4.txt', 'a')
        f.write(f"frame_{count}.jpg, {raw4}, {b-a}/n")
        f.close()
        count+=1
# extractOutput("C:/Users/ADMIN/Desktop/document/junx/Private_Test/videos/scene4cam_05")
output = "C:/Users/ADMIN/Desktop/document/junx/Private_Test/groundtruth/"
path = "C:/Users/ADMIN/Desktop/document/junx/Private_Test/videos/scene4cam_"
for i in range(1, 15):
    fullpath = path+"0"*(2-len(str(i)))+str(i)
    extractOutput(fullpath)