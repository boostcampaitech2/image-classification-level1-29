### 마스크 여부, 성별, 나이를 mapping할 클래스를 생성합니다.
import params
from PIL import Image
import os
import numpy as np
import glob
from torch.utils.data import Dataset,DataLoader 
import pandas as pd
import tqdm

hp = params.Parameters()

class DataFrame():
    
    def __init__(self):
        self.train_dir = hp.data_dir
        self.image_dir = hp.img_dir
        self.labeled_df = self.create_df()
        

    def create_df(self):
        '''==================================================================
        create df with mask status, distinguished with folder of people :) 
        then,it will create labeled dataframe, thanks to our teammate
        ====================================================================='''
        train_df = pd.read_csv(os.path.join(self.train_dir,'train.csv'))
        masks = ['mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'incorrect_mask', 'normal'] #대조할 기준 형성
        wears = ['Wear', 'Wear', 'Wear', 'Wear', 'Wear', 'Incorrect', 'Not Wear']         #라벨링 할 기준 형성
        
        mask_df = pd.DataFrame() #데이터프레임 구조 형성

    
        for person in tqdm.tqdm(train_df.values):
            for mask, wear in zip(masks, wears):
                mask_df=mask_df.append(pd.Series(np.append(person,(mask,wear))),ignore_index=True)
            
                 #한 행의 데이터 프레임간 결합, 
                #컬럼 이름에 맞춰서. ignore_index는 합칠때 인덱스 이름을 무시한 것인 듯 합니다. 

        mask_df.columns = np.append(train_df.columns.values,('mask','wear')) #만든 mask_df에 train_df로부터 컬럼 방향으로 합침, 값과 함께.
                   
        labeled_df = pd.DataFrame()
        
        for index, person in mask_df.itterows(): #pd itterows는 무엇인가? 열에 따라 데이터프레임을 이터레이터처럼 쓸 수 있게 하는건가?
            gender = person['gender']
            gender = 0 if gender == 'male' else 1

            age = person['age']
            age =lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2 #여기 장난으로 랜덤 꼽아도 재미있을듯

            mask = person['mask']
            mask = lambda x:0 if x =='wear' else 1 if x =='incorrect' else 2

        label = 6*mask + 3*gender + age
        labeled_df = labeled_df.append(pd.Series(np.append(person,label)),ignore_index=True)
        labeled_df = labeled_df.astype({'label':int})  #이 부분 지나가면서 한번 더 봐라. 이렇게 하는 것이 옳다.
        

df = DataFrame()
data = df.labeled_df



            


            



        






