import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
import random
import torchsummary
from torchvision import datasets
import matplotlib.pyplot as plt

print(torch.__version__)


batch_size = 64
validation_ratio = 0.1
random_seed = 7993
initial_lr = 0.01
num_epoch = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#==============데이터 적재 파트=====================================================
#데이터 어그멘테이션은 스킵합니다. MNIST는 그 자체로 충분할 터.
transform_train = transforms.Compose([
        #transforms.Resize(32),
        
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)), #특성맵이 바이너리로 한개 뿐이기에, 아래 네트워크를 수정할 자신이 없어 repeat로 동일한 특성맵을 3개로 만들었습니다.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

transform_validation = transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


transform_test = transforms.Compose([
        #transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

trainset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_train)



validset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform_validation)

testset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)

num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(validation_ratio * num_train))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, sampler=train_sampler, num_workers=0
)

valid_loader = torch.utils.data.DataLoader(
    validset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0
)

classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')


#================모듈 파트============================================================================

class ConvReluBatch(nn.Module):                     #DenseNet에 이런 뭉태기가 많이 쓰인다고 해서 가져왔습니다. 
    def __init__(self, nin, nout, kernel_size, stride, padding, bias = False):
        super(ConvReluBatch,self).__init__()
        self.batch_norm = nn.BatchNorm2d(nin)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(nin,nout,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
    def forward(self,x):
        output = self.batch_norm(x)
        output = self.relu(output)
        output = self.conv(output)

        return output



class bottleneck_layer(nn.Sequential):
    def __init__(self, nin, growth_rate, drop_rate=0.2):    
      super(bottleneck_layer, self).__init__()
      
      self.add_module('conv_1x1', ConvReluBatch(nin=nin, nout=growth_rate*4, kernel_size=1, stride=1, padding=0, bias=False))
      self.add_module('conv_3x3', ConvReluBatch(nin=growth_rate*4, nout=growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
      
      self.drop_rate = drop_rate
      
    def forward(self, x):
      bottleneck_output = super(bottleneck_layer, self).forward(x)
      if self.drop_rate > 0:
          bottleneck_output = F.dropout(bottleneck_output, p=self.drop_rate, training=self.training)
          
      bottleneck_output = torch.cat((x, bottleneck_output), 1)
      
      return bottleneck_output
#여기서 torch.cat으로 한번의 bottleneck마다 feature map x가 채널을 따라 누적되는 형태이다. 
#여기서 1은 차원을 의미하며, 1번은 채널 차원을 의미한다.  (Channel-wise로 연산이 진행된다는 뜻.


class TransitionLayer(nn.Sequential): #요걸 사용하면 C, BottleNeck 까지 사용하면 BC. 논문에서는 Compression이라고 칭한다. 하여튼 이번엔 BC모델 구축이다.
    def __init__(self,nin,theta=0.5): #theta는 이 Transition Process의 하이퍼파라미터로서, 1x1 conv의 출력 특성맵 수를 조정한다. 
        super(TransitionLayer,self).__init__()
        self.add_module('1X1 Conv',ConvReluBatch(nin=nin,nout=int(nin*theta),kernel_size=1,stride=1,padding=0,bias=False))
        self.add_module('2x2 AvgPooling', nn.AvgPool2d(kernel_size=2, stride=2, padding=0))



class DenseBlock(nn.Sequential): #단순히 nin과 feature map의 컨트롤용 growth_rate에 따라 bottleneck layer를 쌓아둔 네트워크 형태의 블록.
    def __init__(self, nin, num_bottleneck_layers, growth_rate, drop_rate=0.2):
        super(DenseBlock, self).__init__()

        for i in range(num_bottleneck_layers):
              nin_bottleneck_layer = nin + growth_rate * i
              self.add_module('BottleneckNo_%d' % i, bottleneck_layer(nin=nin_bottleneck_layer, growth_rate=growth_rate, drop_rate=drop_rate))
    
    


class DenseNet(nn.Module):
    def __init__(self, growth_rate =12 , num_layers=100,theta=0.5,drop_rate=0.2,num_classes=10):
        super(DenseNet,self).__init__()
        assert ( num_layers - 4)%6 == 0

        num_bottleneck_layers=(num_layers-4)//6

        self.dense_init = nn.Conv2d(3,growth_rate*2,kernel_size=3,stride=1,padding=1,bias=True)
       # 32 x 32 x (growth_rate*2) --> 32 x 32 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]
        self.dense_block_1 = DenseBlock(nin=growth_rate*2, num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)

        # 32 x 32 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)] --> 16 x 16 x [(growth_rate*2) + (growth_rate * num_bottleneck_layers)]*theta
        nin_transition_layer_1 = (growth_rate*2) + (growth_rate * num_bottleneck_layers) 
        self.transition_layer_1 = TransitionLayer(nin=nin_transition_layer_1, theta=theta)
        
        # 16 x 16 x nin_transition_layer_1*theta --> 16 x 16 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]
        self.dense_block_2 = DenseBlock(nin=int(nin_transition_layer_1*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)

        # 16 x 16 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)] --> 8 x 8 x [nin_transition_layer_1*theta + (growth_rate * num_bottleneck_layers)]*theta
        nin_transition_layer_2 = int(nin_transition_layer_1*theta) + (growth_rate * num_bottleneck_layers) 
        self.transition_layer_2 = TransitionLayer(nin=nin_transition_layer_2, theta=theta)
        
        # 8 x 8 x nin_transition_layer_2*theta --> 8 x 8 x [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)]
        self.dense_block_3 = DenseBlock(nin=int(nin_transition_layer_2*theta), num_bottleneck_layers=num_bottleneck_layers, growth_rate=growth_rate, drop_rate=drop_rate)
        
        nin_fc_layer = int(nin_transition_layer_2*theta) + (growth_rate * num_bottleneck_layers) 
        
        # [nin_transition_layer_2*theta + (growth_rate * num_bottleneck_layers)] --> num_classes
        self.fc_layer = nn.Linear(nin_fc_layer, num_classes)
        
    def forward(self, x):
        dense_init_output = self.dense_init(x)
        
        dense_block_1_output = self.dense_block_1(dense_init_output)
        transition_layer_1_output = self.transition_layer_1(dense_block_1_output)
        
        dense_block_2_output = self.dense_block_2(transition_layer_1_output)
        transition_layer_2_output = self.transition_layer_2(dense_block_2_output)
        
        dense_block_3_output = self.dense_block_3(transition_layer_2_output)
        
        global_avg_pool_output = F.adaptive_avg_pool2d(dense_block_3_output, (1, 1))                
        global_avg_pool_output_flat = global_avg_pool_output.view(global_avg_pool_output.size(0), -1)

        output = self.fc_layer(global_avg_pool_output_flat)
        
        return output

#각 DenseBlock 마다 같은 개수의 convolution 연산을 사용한다. 


def DenseNetBC_100_12():
    return DenseNet(growth_rate=12, num_layers=100, theta=0.5, drop_rate=0.2, num_classes=10)




#================================================================================================================
net = DenseNetBC_100_12()
net.to(device)
#torchsummary.summary(net, (3, 32, 32)) #이게 저희 대회때 자주 사용할만 해 보여요

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[int(num_epoch * 0.5), int(num_epoch * 0.75)], gamma=0.1, last_epoch=-1)

for epoch in range(num_epoch):  
    lr_scheduler.step()
    
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        show_period = 100
        if i % show_period == show_period-1:    # print every "show_period" mini-batches
            print('[%d, %5d/51200] loss: %.7f' %
                  (epoch + 1, (i + 1)*batch_size, running_loss / show_period))
            if (running_loss/show_period)<=0.017:
              break
            running_loss = 0.0
            
        
        
    # validation part
    correct = 0
    total = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('[%d epoch] Accuracy of the network on the validation images: %d %%' % 
          (epoch + 1, 100 * correct / total)
         )
    

print('Finished Training')


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
                
        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))            
            
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i])) 

print ("Done")