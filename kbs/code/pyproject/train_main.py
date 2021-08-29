import torch
from datetime import datetime
from pytz import timezone
from Params import Parameters
import DataFamily
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch.optim as optim
from pprint import pprint as print
from Models import MyModel



'''
we shall execute this file to begin training, 
make inference, and make submission file:)

'''
if __name__ == '__main__':

    hp           = Parameters()                                       #Loading hyperparams...
    dataframe    = DataFamily.DataFrame().dataframe                   #creating dataframe...
    
    trainset     = DataFamily.DataCluster(0,hp.img_dir,dataframe).set #loading trainset...
    validset     = DataFamily.DataCluster(1,hp.img_dir,dataframe).set #loading validset...
    testset      = DataFamily.DataCluster(2,hp.img_dir).set           #loading testset... 

    train_loader = DataLoader(trainset,batch_size=hp.batch_size,num_workers=4,shuffle=True)
    valid_loader = DataLoader(validset,batch_size=hp.batch_size,num_workers=4,shuffle=True) 
    model        = MyModel(hp.model_name,pretrained=True)


    




    
    
    





    

    
    
    #추론은 다른 파일에서? 


    
    
    
    


    
    






