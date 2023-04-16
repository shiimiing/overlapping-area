import pandas
import numpy
import torch
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from torchreid import models

class yolov8:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.model.to('cuda')

    def predict(self, images: Image):
        """
        prediction on cuda
        bounding boxes on cpu

        bounding boxes is a set of bounding boxes of each image.
        """
        prediction = self.model(images, classes = 0, verbose = False)
        bounding_boxes = [predict.boxes.xyxy.to('cpu') for predict in prediction]
        bounding_boxes = torch.stack(bounding_boxes)
        return prediction, bounding_boxes
    
    def crop_image(self, image : Image, bounding_boxes: torch.tensor):
        """
        return list croped image base on image.
        """
        list_image = []
        for box in bounding_boxes:
            box = numpy.array(box)
            new_image = image.crop(box)
            list_image.append(new_image)
        return list_image