from ImportCluster import *

# Set random seed
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore



train_dir = '/opt/ml/input/data/train/images'
csv_dir = '/opt/ml/input/data/train/train.csv'

datalist  = os.listdir(train_dir)






class MaskDataset(Dataset):
    def __init__(self, transforms=None):
        super(MaskDataset,self).__init__()
        self.transforms = transforms
    def __len__(self):
        pass
    def __getitem__(self,idx):
        pass

