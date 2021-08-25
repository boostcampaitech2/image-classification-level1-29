import config
from config import *
import dataset

dataset.get_info()

transform = torchvision.Compose([
    Resize((int(img_height/2), int(img_width/2)),Image.BILINEAR),
    CenterCrop(int(img_height/4),int(img_width/4)),
    RandomHorizontalFlip(0.5),
    RandomRotation(limit=[-45,45]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_set = dataset.TrainDataset(imgs, ans, transform)
data_loader = DataLoader(data_set, batch_size=50, shuffle=True, num_workers=2, drop_last=True)

def get_model():
    model = resnext50_32x4d(pretrained=True)

    model.fc = nn.Linear(2048, CLASS_NUM)

    loss_fn = nn.CrossEntropyLoss()
    optim = optm.Adam(model.parameters(), lr = lr)

    return model.to(device), loss_fn, optim