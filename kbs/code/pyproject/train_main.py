import torch
from datetime import datetime
from pytz import timezone
from params import Parameters
import DataFamily
from torch.utils.data import DataLoader
import pandas as pd
import os
'''
we shall execute this file to begin training, 
make inference, and make submission file:)

'''
if __name__ == '__main__':

    hp = Parameters()                                         #Loading hyperparams...
    dataframe = DataFamily.DataFrame()                        #creating dataframe...
    trainset = DataFamily.DataCluster(0,hp.img_dir,dataframe) #loading trainset...
    validset = DataFamily.DataCluster(1,hp.img_dir,dataframe) #loading validset...
    testset  = DataFamily.DataCluster(2,hp.img_dir)           #loading testset... 

    train_loader = DataLoader(trainset,batch_size=hp.batch_size,num_workers=4,shuffle=True)
    valid_loader = DataLoader(validset,batch_size=hp.batch_size,num_workers=4,shuffle=True) 



    

    #모델
    #훈-련
    #추론은 다른 파일에서? 


    
    
    
    


    
    






