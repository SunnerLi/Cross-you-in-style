import os
import cv2
import json
import torch
import random
import librosa
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as Transform

def getTransform(args):
    resize_m_edge = args.paint_resize_min_edge
    crop_H, crop_W = args.paint_crop_H, args.paint_crop_W

    transform = Transform.Compose([
                Transform.Resize(resize_m_edge),
                Transform.RandomCrop((crop_H, crop_W)),
                # Transform.Resize((crop_H, crop_W)),
                Transform.ToTensor(),
                Transform.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return transform