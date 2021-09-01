import argparse
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from train import addModels, isModelsValid
from mergeSubmissions import merge

InferSplitNum = {
    'all': ['all'],
    'one_by_one': ['mask', 'gender', 'age']
}

def load_model(saved_model, num_classes, device, infer_split):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    if infer_split == 'all':
        model_path = os.path.join(saved_model, 'best.pth')
    else:
        model_path = os.path.join(saved_model, f'best_{infer_split}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    if not isModelsValid(args.models):
        return
    for infer_split, model in zip(InferSplitNum[args.infer_split], args.models.split(',')):
        print(f'Inference splited into {InferSplitNum[args.infer_split]}..inferring -> {infer_split} by {model}')
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        num_classes = MaskBaseDataset.getClassNum(infer_split)  # 18
        model = load_model(model_dir, num_classes, device, infer_split).to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'images')
        info_path = os.path.join(data_dir, 'info.csv')
        info = pd.read_csv(info_path)

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        
        # -- augmentation
        transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
        transform = transform_module(
            resize=args.resize,
            mean=dataset.mean,
            std=dataset.std,
        )
        dataset.set_transform(transform)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

        info['ans'] = preds
        if infer_split == 'all':
            info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
        else:
            info.to_csv(os.path.join(output_dir, f'output_{infer_split}.csv'), index=False)
        print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(384, 384), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--models', type=str, help='input 3 models to infer MASK,GENDER,AGE sequentially(default: args.model,args.model,args.model)')
    parser.add_argument('--infer_split', type=str, default='all', help='choose between [all, one_by_one]')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/one_by_one'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()
    addModels(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
    if args.infer_split == 'one_by_one' and args.models is not None:
        merge()
        print('Merge Done!')