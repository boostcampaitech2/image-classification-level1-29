import config
from config import *

#mask labeling
def get_labels_images(image_dirs : str):
    
    for path in image_dirs:
        for image in glob(f'{path}/**'):
            if not '.-' in image:
                if 'normal' in image:
                    cfg.mask_labels.append(2)
                elif 'incorrect' in image:
                    cfg.mask_labels.append(1)
                else:
                    cfg.mask_labels.append(0)

                cfg.imgs.append(image)


def age_label_func(x):
    if x<30: return 0
    elif 30<=x<60: return 1
    else: return 2


def labeling_gen_age():
    for gender in cfg.df['gender']:
        cfg.gender_labels.extend([gender] * 7)
        
    for age in cfg.df['age']:
        cfg.age_labels.extend([age] * 7)
        

def get_dataFrame():
    df = pd.read_csv(cfg.df_path)

    df['gender'] = df['gender'].map({'female':1, 'male':0})
    df['age'] = df['age'].map(age_label_func)


def get_info():
    image_dirs=[]

    for path in cfg.df.path:
        image_dirs.append(os.path.join(cfg.img_dir, path))

    get_labels_images(image_dirs)
    get_dataFrame()
    labeling_gen_age()

    for idx in range(len(cfg.mask_labels)):
        cfg.ans.append(cfg.mask_labels[idx]*6 + cfg.gender_labels[idx]*3 + cfg.age_labels[idx])
    

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