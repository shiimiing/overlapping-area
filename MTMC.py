import pandas
import numpy
import torch
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import transforms
from PIL import Image, ImageDraw
from ultralytics import YOLO
from torchreid import models

from MOT import yolov8
from ReID import reid

class MTMC:
    def __init__(self):
        self.yolo = yolov8()
        self.ReID = reid()

    def cosine_similarity(self, vectorA : torch.tensor, vectorB : torch.tensor):
        """
        cangculate similarity
        """
        cosi = torch.nn.CosineSimilarity(dim=1)
        return cosi(vectorA,vectorB)[0]