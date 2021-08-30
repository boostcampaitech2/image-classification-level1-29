import pandas as pd
import torch
from pprint import pprint
import os


class Parameters():
    '''data_dir, img_dir, df_path, df, project_name, wandb_dir, and so on....'''
    
    def __init__(self,batch_size=32,num_epoch=20):
        self.data_dir = '/opt/ml/input/data/train'
        self.img_dir = f'{self.data_dir}/images'
        self.test_dir = '/opt/ml/input/data/eval'
        self.submission = pd.read_csv(os.path.join(self.test_dir,'info.csv'))
        self.test_img_dir = os.path.join(self.test_dir, 'images')
        self.test_img_path = [os.path.join(self.test_img_dir, img_id) for img_id in self.submission.ImageID]
        self.df_path = f'{self.data_dir}/train.csv'
        self.df = pd.read_csv(self.df_path)

        self.project_name = 'wandb_bc'
        self.wandb_dir = '/opt/ml/teamrepo/kbs/code/pyproject/model/wandb'

        self.mean = ( 0.5601,0.5241,0.5014)
        self.std = (0.2332,0.2430,0.2456)

        self.img_size = (512,384)
        self.batch_size = batch_size

        self.initial_lr = 0.001
        self.num_epoch = num_epoch
        self.betas = (0.9, 0.999)
        self.weight_decay = 1e-4
        self.tolerance = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = "res2next50"

    pprint('Loading Parameters...')
    
        





