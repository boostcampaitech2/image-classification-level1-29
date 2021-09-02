import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import torch_optimizer as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix
from dataset import MaskBaseDataset
from loss import create_criterion
import warnings
warnings.filterwarnings(action='ignore')


TrainSplitNum = {
    'all': ['all'],
    'one_by_one': ['mask', 'gender', 'age']
}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def conf_mat(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred)
    norm_cm=cm/np.sum(cm, axis=1)[:,None]
    if len(set([y.item() for y in y_true]))==18:
        indices=['wear,m,<30','wear,m,mask<>','wear,m,mask60','wear,f,<30','wear,f,<>','wear,f,60','inc,m,<30','inc,m,<>','inc,m,60','inc,f,<30','inc,f,<>','inc,f,60','nom,m,<30','nom,m,<>','nom,m,60','nom,f,<30','nom,f,<>','nom,f,60']
    else:
        indices=[i for i in range(len(set([y.item() for y in y_true])))]
    cm=pd.DataFrame(norm_cm,index=indices,columns=indices)
    fig=plt.figure(figsize=(11,9))
    sns.heatmap(cm,annot=True)
    return fig

def addModels(args):
    if args.models is None:
        args.models = f'{args.model},{args.model},{args.model}'

def isModelsValid(models):
    models = models.split(',')

    if len(models) < 3:
        print('[ERROR] Require 3 model names with delimiter "," (ex: ResNext50,DenseNet121,EfficientNet_b3)')
        return False

    for model in models:
        try:
            model_module = getattr(import_module("model"), model)
            model_module(num_classes=0)
        except AttributeError:
            print(f'[ERROR] Cannot find model {model}')
            return False
    return True

def rand_bbox(size, lam):
    # reference : https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L279
    W = size[2] 
    H = size[3] 
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)  

   	# uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 세로축으로만 자르기
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(data_dir, model_dir, args):
    if not args.project_split or args.train_split=='all':
        wandb.init(project=args.name, entity='team29',config=config)
        print(args)
    if not isModelsValid(args.models):
        return
    for train_split, model in zip(TrainSplitNum[args.train_split], args.models.split(',')):
        if args.project_split and args.train_split!='all':
            wandb.init(project=args.name+'_'+train_split, entity='team29',config=config)
            print(args)
        print(f'Train splited into {TrainSplitNum[args.train_split]}..training -> {train_split} by {model}')
        seed_everything(args.seed)

        if train_split == 'all':
            save_dir = increment_path(os.path.join(model_dir, args.name))
        else:
            save_dir = os.path.join(model_dir, args.train_split)

        # -- settings
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        
        # -- dataset
        dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
        dataset = dataset_module(
            data_dir=data_dir,
            split=train_split,
        )
        num_classes = dataset.getClassNum(train_split)  # 18

        # -- augmentation
        transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
        transform = transform_module(
            resize=args.resize,
            mean=dataset.mean,
            std=dataset.std,
        )
        dataset.set_transform(transform)

        # -- data_loader
        if args.full_train == "yes":
            train_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=use_cuda,
                drop_last=True,
            )
        else:
            train_set, val_set = dataset.split_dataset()
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=4,
                shuffle=True,
                pin_memory=False,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=args.valid_batch_size,
                num_workers=4,
                shuffle=False,
                pin_memory=False,
                drop_last=True,
            )

        # -- mode
        if train_split == 'all':
            model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        else:
            model_module = getattr(import_module("model"), model)
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        if args.optimizer == "RAdam":
            optimizer = optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        else:
            opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=5e-4
            )
        if args.scheduler == "StepLR":
            scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
        elif args.scheduler == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        elif args.scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 10)
        else:
            raise ValueError

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_val_acc = 0
        best_val_loss = np.inf
        best_val_f1 = 0
        best_train_f1 = 0
        
        # early stopping
        EARLY_STOPPING_PATIENCE = args.patience
        early_stopping_counter = 0

        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            epoch_f1 = 0
            total_pred=torch.tensor([]).to(device)
            total_label=torch.tensor([]).to(device)
            print('Training...')
            with tqdm(train_loader) as pbar:
                for idx, train_batch in enumerate(pbar):
                    inputs, labels = train_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # reference : https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py#L228
                    if args.BETA > 0 and np.random.random() > 0.5: # cutmix가 실행될 경우     
                        lam = np.random.beta(args.BETA, args.BETA)
                        rand_index = torch.randperm(inputs.size()[0]).to(device)
                        target_a = labels # 원본 이미지 label
                        target_b = labels[rand_index] # 패치 이미지 label       
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                        outs = model(inputs)
                        loss = criterion(outs, target_a) * lam + criterion(outs, target_b) * (1. - lam) # 패치 이미지와 원본 이미지의 비율에 맞게 loss를 계산을 해주는 부분
                    else:
                        outs = model(inputs)
                        loss = criterion(outs, labels)

                    preds = torch.argmax(outs, dim=-1)

                    loss.backward()
                    optimizer.step()

                    loss_value += loss.item()
                    matches += (preds == labels).sum().item()
                    iteration_f1 = f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')
                    epoch_f1 += iteration_f1

                    total_pred=torch.hstack((total_pred,preds))
                    total_label=torch.hstack((total_label,labels))

                    train_loss = loss_value / (idx + 1)
                    train_acc = matches / args.batch_size / (idx + 1)
                    train_f1_macro = epoch_f1 / (idx + 1)
                    
                    pbar.set_postfix({'epoch': epoch, 'loss': train_loss, 'acc' : train_acc ,'f1 score': train_f1_macro})
                train_total_pred=total_pred
                train_total_label=total_label
            
            
            if not args.scheduler == 'ReduceLROnPlateau':
                scheduler.step()
            
            if args.full_train == "yes":
                if train_f1_macro > best_train_f1:
                    early_stopping_counter = 0
                    print(f"New best model for f1 score : {train_f1_macro:4.4}! saving the best model..")
                    if train_split == 'all':
                        torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    else:
                        torch.save(model.module.state_dict(), f"{save_dir}/best_{train_split}.pth")
                    best_train_f1 = train_f1_macro
                else:
                    # early stopping
                    early_stopping_counter += 1
                    if early_stopping_counter >= EARLY_STOPPING_PATIENCE:  # patience
                        print("EARLY STOPPING!!")
                        break
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                train_cm=conf_mat(train_total_label.cpu(), train_total_pred.cpu())
                wandb.log({
                    'train loss': train_loss, 'train acc': train_acc, 'train_f1_macro': train_f1_macro, 'train confusion matrix': wandb.Image(train_cm),
                })
                continue

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                loss_value = 0
                matches = 0
                epoch_f1 = 0
                total_pred=torch.tensor([]).to(device)
                total_label=torch.tensor([]).to(device)
                with tqdm(val_loader) as pbar:
                    for idx,val_batch in enumerate(pbar):
                        inputs, labels = val_batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_value += criterion(outs, labels).item()
                        matches += (labels == preds).sum().item()
                        epoch_f1 += f1_score(preds.cpu().numpy(), labels.cpu().numpy(), average='macro')

                        total_pred=torch.hstack((total_pred,preds))
                        total_label=torch.hstack((total_label,labels))

                        val_loss = loss_value / (idx + 1)
                        val_acc = matches / args.batch_size / (idx + 1)
                        val_f1_macro = epoch_f1 / (idx + 1)

                        pbar.set_postfix({'epoch': epoch, 'loss': val_loss, 'acc': val_acc, 'f1 score': val_f1_macro})

                best_val_loss = min(best_val_loss, val_loss)
                best_val_acc = max(best_val_acc, val_acc)
                if val_f1_macro > best_val_f1:
                    early_stopping_counter = 0
                    print(f"New best model for f1 score : {val_f1_macro:4.4}! saving the best model..")
                    if train_split == 'all':
                        torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    else:
                        torch.save(model.module.state_dict(), f"{save_dir}/best_{train_split}.pth")
                    best_val_f1 = val_f1_macro
                else:
                    # early stopping
                    early_stopping_counter += 1
                    if early_stopping_counter >= EARLY_STOPPING_PATIENCE:  # patience
                        print("EARLY STOPPING!!")
                        break
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")

                # confusion matrix
                train_cm=conf_mat(train_total_label.cpu(),train_total_pred.cpu())
                val_cm=conf_mat(total_label.cpu(),total_pred.cpu())
                wandb.log({
                    'train loss': train_loss, 'train acc': train_acc, 'train_f1_macro': train_f1_macro, 'train confusion matrix': wandb.Image(train_cm),
                    'val loss': val_loss, 'val acc': val_acc, 'valid_f1_macro': val_f1_macro, 'val confusion matrix': wandb.Image(val_cm)})
                plt.close()

        if args.project_split and args.train_split!='all':
            wandb.finish()
    if not args.project_split or args.train_split=='all':
        wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # from dotenv import load_dotenv
    import os
    # load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 1)')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience (default: 5)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByClassDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--BETA', type=float, default=-1.0, help="If you want CutMix, give 1.0 (default: -1.0)")
    parser.add_argument("--resize", nargs="+", type=list, default=[224,224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--models', type=str, help='input 3 models to train MASK,GENDER,AGE sequentially(default: args.model,args.model,args.model)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--scheduler', type=str, default='StepLR', help='scheduler type (default: StepLR)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--weight', default=None, help='Input weight if you want')
    parser.add_argument('--train_split', type=str, default='all', help='choose between [all, one_by_one]')
    parser.add_argument('--project_split', type=bool, default=False, help='Set True if you want split when you train models one by one')
    parser.add_argument('--full_train', type=str, default="no", help="If you want full train dataset, give yes (default: no)")

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    addModels(args)
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    config = {
        'seed':args.seed, 'epochs':args.epochs, 'dataset':args.dataset, 'augmentation':args.augmentation,
        'resize':args.resize, 'batch_size':args.batch_size, 'validd_batch_size':args.valid_batch_size,
        'model':args.model, 'models':args.models, 'optimizer':args.optimizer, 'scheduler':args.scheduler, 
        'lr':args.lr, 'val_ratio':args.val_ratio, 'criterion':args.criterion, 'lr_decay_step':args.lr_decay_step, 
        'log_interval':args.log_interval, 'name':args.name, 'weight':args.weight,
        'train_split':args.train_split, 'project_split':args.project_split,
        'data_dir':args.data_dir, 'model_dir':args.model_dir,
        'patience':args.patience, 'BETA':args.BETA, 'full_train':args.full_train
    }
    
    train(data_dir, model_dir, args)
    
