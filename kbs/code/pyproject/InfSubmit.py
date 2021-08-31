from datetime import datetime
import numpy as np
from collections import Counter
import os
import torch
import Params
from pprint import pprint as print
from collections import Counter


from pytz import timezone


hp = Params.Parameters()

class Inference():
    
    def __init__(self,test_loader,model,device):

        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.all_predictions = []
        print("begin guessing...")
        self.result = self.inference(test_loader,model,device,self.all_predictions)
        print("inference done, wish you luck")
        print(Counter(self.result))



    def inference(self,test_loader,model,device,all_predictions):

        for images in enumerate(test_loader):
            with torch.no_grad():
                images = images.to(device)
                scores = model(images)
                preds = scores.argmax(dim=-1)
                all_predictions.extend(preds.cpu().numpy())
        return all_predictions


class Submission():

    def __init__(self,submission):
        self.submission['ans'] = submission
        self.finishedtime = datetime.now(timezone('asia/seoul')).strftime('%Y-%m-%d %H:%M:%S')
        self.csvmaking()
        print("done! please check the counter result, then submit your file.")

    def csvmaking(self,submission):
        submission.to_csv(os.path.join(hp.test_dir, f'{self.finishedtime}.csv'), index=False)




    

        