### 마스크 여부, 성별, 나이를 mapping할 클래스를 생성합니다.
import Params
from PIL import Image
import os
import numpy as np
import glob
from torch.utils.data import Dataset,DataLoader 
import pandas as pd
import tqdm
import pprint as p
from Transformations import TransForm
import albumentations as A
trans=TransForm.transform

print = p.pprint

hp = Params.Parameters()

class DataFrame():
    
    def __init__(self):
        self.train_dir = hp.data_dir
        self.image_dir = hp.img_dir
        self.dataframe = self.create_df()
        

    def create_df(self):
        '''==================================================================
        create df with mask status, distinguished with folder of people :) 
        then,it will create labeled dataframe, thanks to our teammate
        ====================================================================='''
        
        train_df = pd.read_csv(os.path.join(hp.data_dir,'train.csv'))
        masks = ['mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'incorrect_mask', 'normal'] #대조할 기준 형성
        wears = ['Wear', 'Wear', 'Wear', 'Wear', 'Wear', 'Incorrect', 'Not Wear']         #라벨링 할 기준 형성
        mask_df = pd.DataFrame() #데이터프레임 구조 형성
        labeled_df = pd.DataFrame()
        if os.path.exists(os.path.join(hp.data_dir,'labeled_df.csv')):
            print("Dataframe exists, moving on....")
            labeled_df = pd.read_csv(os.path.join(hp.data_dir,'labeled_df.csv'))
            return labeled_df

        else:
            print("first time, it will take some time...")
            if os.path.exists(os.path.join(hp.data_dir,'maskdf.csv')):
                print("dataframe exists, labeling...")
                mask_df=pd.read_csv(os.path.join(hp.data_dir,'maskdf.csv'))
                print("labeling will take some time too.")
                for index, person in mask_df.iterrows(): #pd iterrows는 무엇인가? 열에 따라 데이터프레임을 이터레이터처럼 쓸 수 있게 하는건가?
                    gender = person['gender']
                    gender = 0 if gender == 'male' else 1

                    age = person['age']
                    if age < 30:
                        age = 0
                    elif age >= 30 and age <60:
                        age = 1
                    else:
                        age = 2 #여기 장난으로 랜덤 꼽아도 재미있을듯

                    mask = person['mask']
                    if mask == 'wear':
                        mask = 0
                    elif mask == 'incorrect':
                        mask = 1
                    else:
                        mask = 2


                    label = 6*mask + 3*gender + age
                    labeled_df = labeled_df.append(pd.Series(np.append(person,label)),ignore_index=True)
                labeled_df.columns = np.append(mask_df.columns.values,'label')
                labeled_df = labeled_df.astype({'label':int})  #이 부분 지나가면서 한번 더 봐라. 이렇게 하는 것이 옳다.
                labeled_df.to_csv(os.path.join(hp.data_dir,'labeled_df.csv'))
                return labeled_df


            else:
                print("creating dataframe...it will take about 90 seconds, won't take that long from second time") 
                       
                for person in tqdm.tqdm(train_df.values):
                    for mask, wear in zip(masks, wears):
                        mask_df = mask_df.append(pd.Series(np.append(person, (mask, wear))), ignore_index=True)
                mask_df.columns = np.append(train_df.columns.values, ('mask', 'wear'))
                mask_df.to_csv(os.path.join(hp.data_dir,'maskdf.csv'))
                 #만든 mask_df에 train_df로부터 컬럼 방향으로 합침, 값과 함께.
            print("labeling will take some time too.")


            for index, person in mask_df.iterrows(): #pd iterrows는 무엇인가? 열에 따라 데이터프레임을 이터레이터처럼 쓸 수 있게 하는건가?
                gender = person['gender']
                gender = 0 if gender == 'male' else 1

                age = person['age']
                if age < 30:
                    age = 0
                elif age >= 30 and age <60:
                    age = 1
                else:
                    age = 2


                mask = person['mask']

                if mask == 'wear':
                    mask = 0
                elif mask == 'incorrect':
                    mask = 1
                else:
                    mask = 2


                label = 6*mask + 3*gender + age
                labeled_df = labeled_df.append(pd.Series(np.append(person,label)),ignore_index=True)
            labeled_df.columns=np.append(mask_df.columns.values,'label')
            labeled_df = labeled_df.astype({'label':int})  #이 부분 지나가면서 한번 더 봐라. 이렇게 하는 것이 옳다.
            labeled_df.to_csv(os.path.join(hp.data_dir,'labeled_df.csv'))
            return labeled_df



class DataCluster():
    
    
    def __init__(self,mode,path,labeled_df=None,transform=None):
        '''Creating dataset instance, choose mode to create appropriate dataset 
    mode 0 : Dev, mode 1: Test'''
        self.mode = mode
        self.path = path
        self.labeled_df = labeled_df
        self.transform = transform

        if self.mode == 0 :
            self.set = self.create_train_set(path,labeled_df,transform)
            

        elif self.mode == 1:
            self.set = self.create_valid_set(path,labeled_df,transform)

        elif self.mode == 2:
            self.set = self.create_test_set(path,transform)
        
            
    def create_train_set(self,path,labeled_df,transform):
        print("loading train datasets....")
        transform = trans(0)['train']
        trainset = Devset(path,labeled_df,transform)
        
        return trainset

    def create_valid_set(self,path,labeled_df,transform):
        print("loading valid datasets....")
        transform = trans(1)['val']
        validset = Devset(path,labeled_df,transform)
        
        return validset

    def create_test_set(self,path,transform):
        print("finally, test set...!")
        transform = trans(2)['test']
        testset = TestSet(path,transform)

        return testset
    
    



class Devset(Dataset):
    def __init__(self,path,labeled_df,transform):
        super(Devset).__init__()
        self.path = path
        self.labeled_df = labeled_df
        self.transform = transform
    
    def __getitem__(self, idx):
        full_path = os.path.join(self.path, self.labeled_df.iloc[idx]['path'])
        img_list = glob.glob(full_path + '/*')
        file_name = self.labeled_df.iloc[idx]['mask']
        try:
            image = Image.open(os.path.join(full_path, file_name+'.jpg'))
        except:
            try:
                image = Image.open(os.path.join(full_path, file_name+'.png'))
            except:
                image = Image.open(os.path.join(full_path, file_name+'.jpeg'))
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']



        label = self.labeled_df.iloc[idx]['label']
        return image, label

    def __len__(self):
        return len(self.labeled_df)
    
    def get_labels(self):
        return self.labeled_df['label']



class TestSet(Dataset):
    def __init__(self,img_paths,transform):
        super(TestSet).__init__()
        self.img_paths = hp.test_img_path
        self.transform = transform
        

    def __getitem__(self, index):
        
        image = Image.open(self.img_paths[index])
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image
    
    def __len__(self):
        return len(self.img_paths)

        



            


            



        






