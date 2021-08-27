import os
from PIL import Image
import argparse
from sendMsgToSlack import *

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, models
from torchsummary import summary
from torchensemble import VotingClassifier
from torchensemble.utils.logging import set_logger
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(description='Mask Classification')
parser.add_argument('--input-file', '-i', required=True, type=str, 
                    help='input filename including .csv')
parser.add_argument('--output-file', '-o', required=True, type=str, 
                    help='output filename including .csv')
parser.add_argument('--class-num', '-cn', required=True, type=int, 
                    help='number of classes to classify')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

TRAIN_DIR = '/opt/ml/input/data/train'
TEST_DIR = '/opt/ml/input/data/eval'

TRAIN_EXPAND = args.input_file
SUBMISSION_FILE = args.output_file
CLASS_NUM = args.class_num


class TrainDataset(data.Dataset):
    def __init__(self, img_paths, targets, transform):
        self.img_paths = img_paths
        self.transform = transform
        self.targets = targets

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')
        target = torch.tensor(self.targets[index], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, target.long()

    def __len__(self):
        return len(self.img_paths)

submission = pd.read_csv(os.path.join(TRAIN_DIR, TRAIN_EXPAND))
image_dir = os.path.join(TRAIN_DIR, 'new_imgs')

image_paths = [os.path.join(image_dir, f'{path}/{file}') for path, file in zip(submission.path, submission.file)]
targets = [target for target in submission.target]
transform = transforms.Compose([
    transforms.Resize((223, 223), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])

def get_model():
    model = models.resnext50_32x4d(pretrained=True)
            
    model.fc = nn.Linear(2048, CLASS_NUM)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    return model.to(device), loss_fn, optimizer

model, loss_fn, optimizer = get_model()
schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
print(model)
summary(model, (3,224,224))

# def train(model, optimizer, loss_fn, train_loader):
#     running_loss = 0.0
#     for i, batch in enumerate(iter(train_loader)):
#         x, y = batch
#         model.train()
#         optimizer.zero_grad()
#         prediction = model(x.cuda())
#         loss = loss_fn(prediction, y.cuda().long())
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if i % 100 == 99:
#             print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100}')
#             running_loss = 0.0

# # def validate(model, loss_fn, val_loader):
# #     model.eval()
# #     with torch.no_grad():
# #         val_loss = 0.0
# #         print('Calculating validation results')
# #         for i, batch in enumerate(iter(val_loader)):
# #             inputs, labels = batch
# #             outputs = model(inputs.cuda())
# #             loss = loss_fn(outputs, labels.cuda().long())
# #             val_loss += loss
# #         return val_loss

# dataset = TrainDataset(image_paths, targets, transform)

# # n_val = int(len(dataset) * 0.2)
# # n_train = len(dataset) - n_val
# # train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])

# train_loader = data.DataLoader(dataset, batch_size=50, shuffle=True, num_workers=1, drop_last=True)
# # val_loader = data.DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=2, drop_last=True)

# for epoch in range(10):
#     print(f" epoch {epoch + 1}/10")
#     train(model, optimizer, loss_fn, train_loader)
#     # val_loss = validate(model, loss_fn, val_loader)    
#     # schedular.step(val_loss)


#### ensenble test ####
dataset = TrainDataset(image_paths, targets, transform)

train_loader = data.DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2, drop_last=True)

logger = set_logger('classification_mask_mlp')

model = VotingClassifier(estimator=model, n_estimators=10, cuda=True)
model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
model.fit(
    train_loader,
    epochs=10
)
#########################


submission = pd.read_csv(os.path.join(TEST_DIR, 'info.csv'))
image_dir = os.path.join(TEST_DIR, 'new_imgs')

image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    transforms.Resize((224, 224), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])

class TestDataset(data.Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

dataset = TestDataset(image_paths, transform)

loader = data.DataLoader(
    dataset,
    shuffle=False
)

all_predictions = []
model.eval()
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        # pred = model(images)
        pred = model.predict(images) # ensenble test
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

submission.to_csv(os.path.join(TEST_DIR, SUBMISSION_FILE), index=False)
print('test inference is done!')
send_message_to_slack(SUBMISSION_FILE)