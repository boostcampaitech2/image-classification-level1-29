import torch.nn.functional as F
from sklearn.metrics import f1_score
import torch.optim as optim
from pprint import pprint as print
import warnings
from torch.optim import optimizer
from Models import MyModel
from tqdm import tqdm
import torch
from Params import Parameters
import numpy as np
import wandb
import time
warnings.filterwarnings(action='ignore')

hp = Parameters()


class TrainandEval():


    def __init__(self,num_epoch,batch_size,train_loader,valid_loader,model,optimizer,lr_scheduler):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        model = model.to(hp.device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epoch = num_epoch
        self.batch_size=batch_size
        self.best_model = self.learn(optimizer)
        



    def learn(self,optimizer):
        config={'epochs':self.num_epoch,'batch_size':self.batch_size,'learning_rate':hp.initial_lr}
        wandb.init(project='pyproject',config=config,dir=hp.wandb_dir)
        valid_best_loss = float('inf')
        optimizer = optimizer
        for e in range(self.num_epoch):
            print(f'=====================epoch %d========================' % (e+1))
            train_loss_list = []
            train_acc_list = []
            train_epoch_f1 = 0
            n_iter = 0
            
            self.model.train()
            since = time.time()

            for i, (images,targets) in enumerate(tqdm(self.train_loader)):

                self.optimizer.zero_grad()

                images  = images.to(hp.device)
                targets = targets.to(hp.device)

                scores  = self.model(images)
                _,preds   = scores.max(dim=1)
                
                loss = F.cross_entropy(scores,targets)
                loss.backward()
                self.optimizer.step()

                correct = sum(targets == preds).cpu()
                acc = (correct/images.shape[0])*100
                
                train_epoch_f1 += f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average = 'macro')
                n_iter += 1

                train_acc_list.append(acc)
                train_loss_list.append(loss)
                wandb.log({'train loss' : np.sum(train_loss_list)/(i+1), 'train_accuracy' : (np.sum(train_acc_list)/(i+1)),
            'train_f1 score':f1_score(targets.cpu(),preds.cpu(),average='weighted')})
                if i % 50 == 0 :
                    #print(f'Iteration %3.d | Train Loss  %.4f | Classifier Accuracy %2.2f' % (i, loss, acc))
                    pass
                
            train_mean_loss = np.mean(train_loss_list, dtype = "float64")
            train_mean_acc  = np.mean(train_acc_list,  dtype = "float64")
            train_epoch_f1 = train_epoch_f1/n_iter
            
            epoch_time = time.time() - since
            

            print('')
            print(f'[Summary] Elapsed time : %.0f m %.0f s' % (epoch_time // 60, epoch_time % 60))
            print(f'Train Loss Mean %.4f | Accuracy %2.2f | F1-Score %2.4f' % (train_mean_loss, train_mean_acc, train_epoch_f1))

            self.model.eval()
            valid_epoch_f1 = 0
            valid_n_iter = 0
            valid_loss_list = []
            valid_acc_list = []

            for i, (images,targets) in enumerate(tqdm(self.valid_loader)):
                optimizer.zero_grad()
                images  = images.to(hp.device)
                targets = targets.to(hp.device)

                with torch.no_grad():
                    scores = self.model(images)
                    loss = F.cross_entropy(scores,targets)
                    _,preds = scores.max(dim=1)
                    valid_epoch_f1 += f1_score(preds.cpu().numpy(),targets.cpu().numpy(),average='weighted')
                    valid_n_iter +=1
                correct = sum(targets==preds).cpu()
                acc = (correct/images.shape[0])*100
                valid_loss_list.append(loss)
                valid_acc_list.append(acc)
                wandb.log({'valid loss' : np.sum(valid_loss_list)/(i+1), 'valid_accuracy' : (np.sum(valid_acc_list)/(i+1)),
            'valid_f1 score':f1_score(targets.cpu(),preds.cpu(),average='weighted')})
            val_mean_loss = np.mean(valid_loss_list, dtype="float64")
            val_mean_acc = np.mean(valid_acc_list, dtype="float64")
            valid_epoch_f1 = valid_epoch_f1/n_iter
    
            print(f'Valid Loss Mean %.4f | Accuracy %2.2f | F1-Score %2.4f' % (val_mean_loss, val_mean_acc, valid_epoch_f1) )
            print('')

            if val_mean_loss < valid_best_loss:
                valid_best_loss = val_mean_loss
                valid_early_stop = 0
                # new best model save (valid 기준)
                best_model = self.model
                # path = './model/'
                # torch.save(best_model.state_dict(), f'{path}model{val_mean_acc:2.2f}_epoch_{e}.pth')
            else:
                # early stopping    
                valid_early_stop += 1
                if valid_early_stop >= hp.tolerance:
                    print("EARLY STOPPING!!")
                    return best_model
                    

            self.lr_scheduler.step()
        return best_model



            
            

