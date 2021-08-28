import os, glob
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from torchvision.models import resnet50


import torch_optimizer as optim

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('using device:', device)


model = resnet50(pretrained=True, progress=False)
print(model)

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(MyModel, self).__init__()
        self.model = resnet50(pretrained=True, progress=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.model.fc(x)
        x = self.model.fc(x)
        
        return x

mask_model = MyModel(num_classes=3)
for param in mask_model.parameters():
    param.requires_grad = False # frozen
for param in mask_model.model.fc.parameters():
    param.requires_grad = True # 마지막 레이어 살리기
mask_model.to(device)


gender_model = MyModel(num_classes=2)
for param in gender_model.parameters():
    param.requires_grad = False # frozen
for param in gender_model.model.fc.parameters():
    param.requires_grad = True # 마지막 레이어 살리기
gender_model.to(device)


age_model = MyModel(num_classes=3)
for param in age_model.parameters():
    param.requires_grad = False # frozen
for param in age_model.model.fc.parameters():
    param.requires_grad = True # 마지막 레이어 살리기
age_model.to(device)