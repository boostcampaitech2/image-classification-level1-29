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
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop
import albumentations as A
from albumentations.pytorch import ToTensorV2

class cfg:
    data_dir = "/opt/ml/input/data/train/"
    img_dir = f"{data_dir}/images"
    df_path = f"{data_dir}/train.csv"
    img_height = 512
    img_width = 384
    batch_size = 128
    lr = 0.001
    epoch = 10

    mask_labels=[]
    age_labels=[]
    gender_labels=[]
    imgs=[]
    ans = []
    df = pd.read_csv(df_path)