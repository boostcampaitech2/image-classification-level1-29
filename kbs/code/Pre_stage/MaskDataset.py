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
from sklearn.model_selection import train_test_split
from torchvision.transforms import Resize, ToTensor, Normalize
from torchvision import transforms, utils

train_dir = '/opt/ml/input/data/train'
trainimage_dir = os.path.join(train_dir, 'images')


# meta 데이터와 이미지 경로를 불러옵니다.
train_df = pd.read_csv(os.path.join(train_dir, 'train.csv'))

masks = ['mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'incorrect_mask', 'normal']
wears = ['Wear', 'Wear', 'Wear', 'Wear', 'Wear', 'Incorrect', 'Not Wear']
mask_df = pd.DataFrame()
for person in train_df.values:
    for mask, wear in zip(masks, wears):
        mask_df = mask_df.append(pd.Series(np.append(person, (mask, wear))), ignore_index=True)
mask_df.columns = np.append(train_df.columns.values, ('mask', 'wear'))
mask_df = mask_df.sample(frac=1).reset_index(drop=True)

train, valid = train_test_split(mask_df, test_size=0.2, stratify=mask_df['wear'])
transform = transforms.Compose([
    Resize((224, 224), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])


class GenderDataset(Dataset):
    def __init__(self, path, mask_df, transform):
        super(GenderDataset).__init__()
        self.path = path
        self.mask_df = mask_df
        self.transform = transform
        
    def __getitem__(self, idx):
        full_path = os.path.join(self.path, self.mask_df.iloc[idx]['path'])
        img_list = glob.glob(full_path + '/*')
        file_name = self.mask_df.iloc[idx]['mask']
        for img_name in img_list:
            if img_name.startswith(file_name):
                break
        image = Image.open(os.path.join(full_path, img_name))
        if self.transform:
            image = self.transform(image)
        
        label = self.mask_df.iloc[idx]['gender']
        label = 0 if label=='male' else 1
        return image, label
    
    def __len__(self):
        return len(self.mask_df)


gender_train_data = GenderDataset(trainimage_dir, train, transform)
gender_valid_data = GenderDataset(trainimage_dir, valid, transform)

gender_train = DataLoader(gender_train_data, batch_size=32, shuffle=True, num_workers=4)
gender_valid = DataLoader(gender_valid_data, batch_size=32, shuffle=True, num_workers=4)

class AgeDataset(Dataset):
    def __init__(self, path, mask_df, transform):
        super(AgeDataset).__init__()
        self.path = path
        self.mask_df = mask_df
        self.transform = transform
        
    def __getitem__(self, idx):
        full_path = os.path.join(self.path, self.mask_df.iloc[idx]['path'])
        img_list = glob.glob(full_path + '/*')
        file_name = self.mask_df.iloc[idx]['mask']
        for img_name in img_list:
            if img_name.startswith(file_name): #여길 주목하자.
                break
        image = Image.open(os.path.join(full_path, img_name))
        if self.transform:
            image = self.transform(image)
        
        label = self.mask_df.iloc[idx]['age']
        if label >= 60.0:
            label = 2
        elif label >= 30.0:
            label = 1
        else:
            label = 0
        return image, label
    
    def __len__(self):
        return len(self.mask_df)

age_train_data = AgeDataset(trainimage_dir, train, transform)
age_valid_data = AgeDataset(trainimage_dir, valid, transform)

age_train = DataLoader(age_train_data, batch_size=32, shuffle=True, num_workers=4)
age_valid = DataLoader(age_valid_data, batch_size=32, shuffle=True, num_workers=4)

class MaskDataset(Dataset):
    def __init__(self, path, mask_df, transform):
        super(MaskDataset).__init__()
        self.path = path
        self.mask_df = mask_df
        self.transform = transform
        
    def __getitem__(self, idx):
        full_path = os.path.join(self.path, self.mask_df.iloc[idx]['path'])
        img_list = glob.glob(full_path + '/*')
        file_name = self.mask_df.iloc[idx]['mask']
        for img_name in img_list:
            if img_name.startswith(file_name):
                break
        image = Image.open(os.path.join(full_path, img_name))
        if self.transform:
            image = self.transform(image)
        
        label = self.mask_df.iloc[idx]['mask']
        if label.startswith('mask'):
            label = 0
        elif label.startswith('incorrect'):
            label = 1
        else:
            label = 2
        return image, label
    
    def __len__(self):
        return len(self.mask_df)

mask_train_data = MaskDataset(trainimage_dir, train, transform)
mask_valid_data = MaskDataset(trainimage_dir, valid, transform)

mask_train = DataLoader(mask_train_data, batch_size=32, shuffle=True)
mask_valid = DataLoader(mask_valid_data, batch_size=32, shuffle=True)


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        super(TestDataset).__init__()
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


test_dir = '/opt/ml/input/data/eval'

submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
testimage_dir = os.path.join(test_dir, 'images')
# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(testimage_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    Resize((224, 224), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
test_dataset = TestDataset(image_paths, transform)

test_loader = DataLoader(
    test_dataset,
    shuffle=False
)