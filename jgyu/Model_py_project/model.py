import config
from config import *
import dataset
import loss

dataset.get_info()

transform = transforms.Compose([
    Resize((int(IMG_HEIGHT/2), int(IMG_WIDTH/2)),Image.BILINEAR),
    CenterCrop(int(IMG_WIDTH/4)),
    RandomHorizontalFlip(0.5),
    RandomRotation(degrees=[-45,45]),
    ColorJitter(brightness=0.5, hue=0.3),
    GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5)),
    RandomInvert(),
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
    model = densenet161(pretrained=True)
    
    model.classifier = nn.Linear(model.classifier.in_features, CLASS_NUM)

    loss_fn = loss.LabelSmoothingLoss(CLASS_NUM, 0.2)
    optim = optm.Adam(model.parameters(), lr = LR)

    return model.to(device), loss_fn, optim