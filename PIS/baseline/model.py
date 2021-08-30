import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class AgeAffectedByGender(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mask=timm.create_model('efficientnet_b3a', pretrained=True, num_classes=3)
        self.gender=timm.create_model('efficientnet_b3a', pretrained=True, num_classes=2)
        self.mage=timm.create_model('efficientnet_b3a', pretrained=True, num_classes=3)
        self.fage=timm.create_model('efficientnet_b3a', pretrained=True, num_classes=3)

    def forward(self, x):
        mask=nn.Softmax(dim=1)(self.mask(x))
        gender=nn.Softmax(dim=1)(self.gender(x))
        age=nn.Softmax(dim=1)(gender[:,0]*self.mage(x)+gender[:,1]*self.fage(x))
        return (mask.view(x.size(0),3,1,1)*gender.view(x.size(0),1,2,1)*age.view(x.size(0),1,1,3)).view(x.size(0),-1)
