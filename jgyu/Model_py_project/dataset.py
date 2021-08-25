import config
from config import *

def get_labels_images(image_dirs : str):
    
    for path in image_dirs:
        for image in glob(f'{path}/**'):
            if not '.-' in image:
                if 'normal' in image:
                    mask_labels.append(2)
                elif 'incorrect' in image:
                    mask_labels.append(1)
                else:
                    mask_labels.append(0)

                imgs.append(image)

def labeling_gen_age():
    for gender in df['gender']:
        gender_labels.extend([gender] * 7)
        
    for age in df['age']:
        age_labels.extend([age] * 7)
        

def get_dataFrame():
    df['gender'] = df['gender'].map({'female': 1, 'male': 0})
    df['age'] = df['age'].map(lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2)


def get_info():
    image_dirs=[]

    get_dataFrame()

    for path in df.path:
        image_dirs.append(os.path.join(IMG_DIR, path))
    
    get_labels_images(image_dirs)
    labeling_gen_age()


    for idx in range(len(mask_labels)):
        ans.append(int(mask_labels[idx])*6 + int(gender_labels[idx])*3 + int(age_labels[idx]))
    

class TrainDataset(Dataset):
    def __init__(self, img_paths, labels, transform):
        self.img_paths = img_paths
        self.transform = transform
        self.labels = labels

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def __len__(self):
        return len(self.img_paths)

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