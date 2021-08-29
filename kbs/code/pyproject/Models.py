import timm
from timm.models.layers.classifier import ClassifierHead
import torch.nn as nn




class MyModel(nn.Module):
    def __init__(self,model_name,pretrained=True):
        super().__init__()

        self.model=timm.create_model(model_name,pretrained=True)
        n_features=self.model.num_features
        self.mask_classifier=ClassifierHead(n_features,3)
        self.gender_classifier=ClassifierHead(n_features,2)
        self.age_classifier=ClassifierHead(n_features,3)

    def forward(self,x):

        x=self.model.forward_features(x)
        mask=self.mask_classifier(x).view(x.size(0),3,1,1)
        gender=self.gender_classifier(x).view(x.size(0),1,2,1)
        age=self.age_classifier(x).view(x.size(0),1,1,3)

        return (mask*gender*age).view(x.size(0),-1)