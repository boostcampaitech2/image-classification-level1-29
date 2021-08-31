import timm
import torch
import torchvision
import torch.nn as nn

from timm.models.layers.classifier import ClassifierHead

#timm 쓰기 편해보여서 가져왔습니다

model = timm.create_model('res2next50', pretrained=True)

class MyModel(nn.Module):
    def __init__(self,model_name,pretrained=True):
        super().__init__()
        self.model=timm.create_model(model_name,pretrained=True)
        n_features=self.model.num_features
        self.mask_result=ClassifierHead(n_features,3)
        self.gender_result=ClassifierHead(n_features,2)
        self.age_result=ClassifierHead(n_features,3)
    def forward(self,x):
        x=self.model.forward_features(x)
        age_label=self.age_result(x)
        gender_label=self.gender_result(x)
        mask_label=self.mask_result(x)
        
        return age_label,gender_label,mask_label