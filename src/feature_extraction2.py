import cv2
import numpy as np
import os
from pathlib import Path
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class_map = 
{
    0: 'glass',
    1: 'paper',
    2: 'cardboard',
    3: 'plastic',
    4: 'metal',
    5: 'trash'
}

device = torch.device("cpu")
model = models.resnet50(pretrained=True)
featureExtractor = torch.nn.Sequential(*list(model.children())[:-1])
featureExtractor = featureExtractor.to(device)
featureExtractor.eval()

preprocess = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

def extract_cnn_features(imagePath):
    try:
        image = Image.open(imagePath).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not load image {imagePath}: {e}")
    inputTensor = preprocess(image)
    inputBatch = inputTensor.unsqueeze(0)
    inputBatch = inputBatch.to(device)

