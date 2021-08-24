import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchsummary import summary
import os
import pandas as pd
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dir = '/opt/ml/input/data/train'

class TrainDataset(Dataset):
    def __init__(self, img_paths, targets, transform):
        self.img_paths = img_paths
        self.transform = transform
        self.targets = targets

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        target = torch.tensor(self.targets[index], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.img_paths)

submission = pd.read_csv(os.path.join(train_dir, 'train_expand_sex.csv'))
# submission = pd.read_csv(os.path.join(train_dir, 'train_expand_mask.csv'))
# submission = pd.read_csv(os.path.join(train_dir, 'train_expand_age.csv'))
image_dir = os.path.join(train_dir, 'images')

image_paths = [os.path.join(image_dir, f'{path}/{file}') for path, file in zip(submission.path, submission.file)]
targets = [target for target in submission.target]
transform = transforms.Compose([
    transforms.Resize((512, 384), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
dataset = TrainDataset(image_paths, targets, transform)
train_loader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)

def train_batch(x, y, model, opt, loss_fn):
    model.train()
    prediction = model(x) 
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()

def get_model():
    model = models.resnext50_32x4d(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
            
    model.fc = nn.Linear(2048, 2)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    return model.to(device), loss_fn, optimizer

model, loss_fn, optimizer = get_model()
# print(model)
# summary(model, (3,512,384))

for epoch in range(2):
    print(f" epoch {epoch + 1}/2")

    for ix, batch in enumerate(iter(train_loader)):
        x, y = batch
        batch_loss = train_batch(x.cuda(), y.cuda().long(), model, optimizer, loss_fn)


test_dir = '/opt/ml/input/data/eval'

submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    transforms.Resize((512, 384), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

dataset = TestDataset(image_paths, transform)

loader = DataLoader(
    dataset,
    shuffle=False
)

device = torch.device('cuda')
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

submission.to_csv(os.path.join(test_dir, 'submission_sex.csv'), index=False)
# submission.to_csv(os.path.join(test_dir, 'submission_mask.csv'), index=False)
# submission.to_csv(os.path.join(test_dir, 'submission_age.csv'), index=False)
print('test inference is done!')