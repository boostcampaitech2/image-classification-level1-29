import enum
import torch
from datetime import datetime
from pytz import timezone
from Params import Parameters
from DataFamily import DataCluster,DataFrame,DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import Adam
from sklearn.metrics import f1_score
from pprint import pprint as print
from Models import MyModel
from tqdm import tqdm
from Trainer import TrainandEval
from torch.utils.data import dataset
from InfSubmit import Inference, Submission




'''
we shall execute this file to begin training, 
make inference, and make submission file:)

'''
if __name__ == '__main__':

    hp           = Parameters()                                             #Loading hyperparams...
    dataframe    = DataFrame().dataframe                                    #creating dataframe...
            
    trainset     = DataCluster(0,hp.img_dir,dataframe).set                  #loading trainset...
    validset     = DataCluster(1,hp.img_dir,dataframe).set                  #loading validset...
    testset      = DataCluster(2,hp.test_img_path).set                      #loading testset... 

    train_loader = DataLoader(trainset,batch_size=hp.batch_size,num_workers=4,shuffle=True)
    valid_loader = DataLoader(validset,batch_size=hp.batch_size,num_workers=4,shuffle=True) 
    
    model        = MyModel(hp.model_name,pretrained=True)                   #loading model...
    optimizer    = Adam(model.parameters(),lr=hp.initial_lr,weight_decay=hp.weight_decay)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100,T_mult=1, eta_min=0.00001)

    train        = TrainandEval(20,32,train_loader,valid_loader,model,optimizer,lr_scheduler)

    modelresult  = train.best_model
    test_loader  = DataLoader(testset,shuffle=False,num_workers=4)
    inference    = Inference(test_loader,modelresult,hp.device)
    submission   = Submission(inference)





    

    




        



    




    
    
    





    

    
    
    #추론은 다른 파일에서? 


    
    
    
    


    
    






