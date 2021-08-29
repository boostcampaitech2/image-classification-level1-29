import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms
import Params
hp = Params.Parameters()

class TransForm():
    
    def __init__(self,mode,mean=hp.mean,std=hp.std):
        '''mode 0 is for trainset, mode 1 is for validset, mode 2 is for testset.'''
        
        self.mode = mode
        self.mean = mean
        self.std = std


    def transform(mode=0,mean=hp.mean,std=hp.std,img_size=hp.img_size):

        transformations = {}
        if mode == 0 :
            transformations['train'] = A.Compose([
            Resize(img_size[0], img_size[1], p=1.0),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            GaussNoise(p=0.5),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),], p=1.0)

        elif mode == 1:
            transformations['val'] = A.Compose([
            Resize(img_size[0], img_size[1]),
            Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),], p=1.0)
        
        elif mode == 2:
            transformations['test'] = A.Compose([ToTensorV2(p=1.0)],p=1.0)
        return transformations
            
        
        


        