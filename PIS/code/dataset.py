import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

cwd=os.path.dirname(os.getcwd())

def make_images(meta,img_dir,train):
    images=[]
    labels=[]
    if train:
        for idx in range(len(meta)):
            folder_path=os.path.join(img_dir, meta.path[idx])
            for img in os.listdir(folder_path):
                if '._' in img or '.ipynb' in img:
                    continue
                images.append(os.path.join(folder_path,img))
                labels.append((('incorrect' in img)+('normal' in img)*2)*6+(meta.gender[idx]=='female')*3+(30<=meta.age[idx])+(60<=meta.age[idx]))
    else:
        for img_id in meta.ImageID:
            images.append(os.path.join(img_dir, img_id))
    return images,labels

trans=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(Dataset):
    def __init__(self,transform=trans,train=True):
        self.train=train
        self.md=['info','train']
        self.path=[os.path.join(cwd,'input/data/eval'),os.path.join(cwd,'input/data/train')]
        self.meta=pd.read_csv(os.path.join(self.path[train], f'{self.md[train]}.csv'))
        self.img_dir=os.path.join(self.path[train],'images')
        self.classes=[('Wear','Incorrect','Not Wear'),('남','여'),('<30','>=30 and <60','>=60')]
        self.trans=trans
        
        self.images,self.labels=make_images(self.meta,self.img_dir,train)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image=Image.open(self.images[idx])
        image=self.trans(image)
        if train:
            label=self.labels[idx]
        else:
            label=0
        return image,label
