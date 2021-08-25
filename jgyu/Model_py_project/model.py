import config
from config import *
import dataset

dataset.get_info()

transform = transforms.Compose([
    Resize((int(IMG_HEIGHT/2), int(IMG_WIDTH/2)),Image.BILINEAR),
    CenterCrop(int(IMG_WIDTH/4)),
    RandomHorizontalFlip(0.5),
    RandomRotation(degrees=[-45,45]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_set = dataset.TrainDataset(imgs, ans, transform)
data_loader = DataLoader(data_set, batch_size=50, shuffle=True, num_workers=2, drop_last=True)

submission = pd.read_csv(os.path.join(TEST_DIR, 'info.csv'))
image_dir = os.path.join(TEST_DIR, 'images')
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

test_data_set = dataset.TestDataset(image_paths, transform)
test_data_loader = DataLoader(test_data_set, shuffle=False)

def get_model():
    model = resnext50_32x4d(pretrained=True)

    model.fc = nn.Linear(2048, CLASS_NUM)

    loss_fn = nn.CrossEntropyLoss()
    optim = optm.Adam(model.parameters(), lr = LR)

    return model.to(device), loss_fn, optim