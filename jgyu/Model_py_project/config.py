import torch
import torchvision
from torchvision import transforms
import PIL
from PIL import Image
import os
import glob
import cv2
from glob import glob
import pandas as pd
import numpy as np
import torch.optim as optm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop, RandomHorizontalFlip, RandomRotation
from torchvision.models import resnext50_32x4d
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

CLASS_NUM=18

DATA_DIR = "/opt/ml/input/data/train/"
IMG_DIR = f"{DATA_DIR}/images"
DF_PATH = f"{DATA_DIR}/train.csv"
TEST_DIR = "/opt/ml/input/data/eval/"

IMG_HEIGHT = 512
IMG_WIDTH = 384
BATCH_SIZE = 128
LR = 0.001
EPOCH = 10

mask_labels=[]
age_labels=[]
gender_labels=[]
imgs=[]
ans = []
    
df = pd.read_csv(DF_PATH)