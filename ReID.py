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

#prepocess image.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class reid:
    def __init__(self):
        self.model = models.build_model('resnet50', 75, loss='softmax')
        self.model.to('cuda')

    def preprocess(self, image):
        return preprocess(image).to('cuda').unsqueeze(0)
    
    def get_features_vector(self, images: torch.tensor):
        """
        input is a batch of preprocess image
        with a data type torch.tensor.
        """
        return self.model(images)