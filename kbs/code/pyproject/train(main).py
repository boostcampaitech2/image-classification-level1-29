import torch
from datetime import datetime
from pytz import timezone
from params import Parameters
import datafamily
import pandas as pd
import os
'''
we shall execute this file to begin training, 
make inference, and make submission file:)
'''
if __name__ == '__main__':

    hp = Parameters(3) #Loading hyperparams...
    
    dataframe = datafamily.DataFrame()

    devset = datafamily.DataCluster(0,hp.img_dir,dataframe)
    testset = datafamily.DataCluster(1,hp.img_dir)


    
    
    
    


    
    






