import argparse
import glob
import json
import os
import random
import re
from importlib import import_module
from pathlib import Path
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from dataset import MaskBaseDataset
from loss import create_criterion
import warnings
warnings.filterwarnings(action='ignore')


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


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    tolerance = 10
    valid_early_stop = 0
    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        total_pred=torch.tensor([]).to(device)
        total_label=torch.tensor([]).to(device)
        with tqdm(train_loader) as pbar:
            for idx, train_batch in enumerate(pbar):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                total_pred=torch.hstack((total_pred,preds))
                total_label=torch.hstack((total_label,labels))
                '''if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0'''
                train_loss=loss_value/(idx+1)
                train_acc=matches/args.batch_size/(idx+1)
                pbar.set_postfix({'epoch' : epoch, 'loss' :train_loss, 'accuracy' : train_acc ,'F1 score':f1_score(total_label.cpu(),total_pred.cpu(),average='weighted')})
        
<<<<<<< Updated upstream
=======
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

        if args.full_train == 'yes':
            if train_split == 'all':
                model_module = getattr(import_module("model"),args.model)
            
                model = model_module(num_classes = num_classes)
                check = torch.load(model_dir)
                model.load_state_dict(check['model_state_dict'],strict=False)
                model = model.to(device)

                optimizer=optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
                optimizer.load_state_dict(check['optimizer_state_dict'])
        else:
            print("we are in the else")    
        # -- mode
            if train_split == 'all':
                model_module = getattr(import_module("model"), args.model)  # default: BaseModel
            else:
                model_module = getattr(import_module("model"), model)
            model = model_module(
                num_classes=num_classes
            ).to(device)
            # model = torch.nn.DataParallel(model)

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
>>>>>>> Stashed changes
        

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            #figure = None
            total_pred=torch.tensor([]).to(device)
            total_label=torch.tensor([]).to(device)
            with tqdm(val_loader) as pbar:
                for idx,val_batch in enumerate(pbar):
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    total_pred=torch.hstack((total_pred,preds))
                    total_label=torch.hstack((total_label,labels))

<<<<<<< Updated upstream
                    '''if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                        )'''
                    pbar.set_postfix({'tol':valid_early_stop,'epoch' : epoch, 'loss' :np.sum(val_loss_items)/len(val_loss_items), 'accuracy' : np.sum(val_acc_items)/len(val_set),'F1 score':f1_score(total_label.cpu(),total_pred.cpu(),average='weighted')})
=======
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
                torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, f"{save_dir}/last.pth")
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
>>>>>>> Stashed changes

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
                valid_early_stop=0
            else:
                valid_early_stop+=1
                if valid_early_stop >= tolerance:  # patience
                    print("EARLY STOPPING!!")
                    break

            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            '''print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)'''
                    
            wandb.log({'train loss': train_loss, 'train acc' : train_acc,'val loss' : val_loss, 'val acc' :val_acc })
    wandb.finish()


if __name__ == '__main__':
    
    

    parser = argparse.ArgumentParser()

    #from dotenv import load_dotenv
    import os
    #load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[224,224], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
<<<<<<< Updated upstream

=======
    parser.add_argument('--weight', default=None, help='Input weight if you want')
    parser.add_argument('--train_split', type=str, default='all', help='choose between [all, one_by_one]')
    parser.add_argument('--project_split', type=bool, default=False, help='Set True if you want split when you train models one by one')
    parser.add_argument('--full_train', type=str, default="no", help="If you want full train dataset, give yes (default: no)")
 
>>>>>>> Stashed changes
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    config = {'epochs':args.epochs,'batch_size':args.batch_size,'learning_rate':args.lr}
    wandb.init(project='-', entity='team29',config=config)
    train(data_dir, model_dir, args)