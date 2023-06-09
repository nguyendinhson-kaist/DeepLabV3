import torch 
import torchvision
from torch.utils.data import DataLoader, Subset

import numpy as np
import random

from dataset import *
from utils import *
from modules import *

import argparse
import sys
import thop

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('output_stride', type=int, choices=[8,16],
        help='output stride of model')

    parser.add_argument('--use-resnet101', action='store_true',
        help='use resnet101 instead of resnet50')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
        help='the learning rate')
    parser.add_argument('-e', '--num-epochs', type=int, default=5, 
        help='the number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=8,
        help='the batch size for training and validation')
    parser.add_argument('--use-subset', action='store_true',
        help='use subset to sanity check')
    parser.add_argument('--trainaug', action='store_true', 
        help='use trainaug dataset for training')
    parser.add_argument('--gpu-id', type=int, default=0,
        help='select gpu if there are many gpus, be sure id is valid')
    parser.add_argument('--seed', type=int, default=1,
        help='set random seed')

    parser.add_argument('--log-every', type=int, default=25,
        help='print and log loss info at every iter')
    parser.add_argument('-v', '--verbose', action='store_true', 
        help='print info')

    args = parser.parse_args()

    return args


def main():
    # get input arguments
    args = parse_args()

    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    print("Using",device)

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # prepare dataset
    DATASET_PATH = 'dataset/'

    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 
                    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


    train_preprocess = CoCompose([
        CoRandomResized(scale=(0.5, 2.0)),
        CoRandomFlip,
        CoRandomCrop,
        CoToTensor,
        CoNormalize
    ])

    val_preprocess = CoCompose([
        CoResize,
        CoCenterCrop,
        CoToTensor,
        CoNormalize
    ])

    mask_dataset = range(100)

    if args.trainaug:
        full_train_dataset = VOCSegmentation(root=DATASET_PATH+'train', year="2012_aug", image_set="train", download=False, transforms=train_preprocess)
    else:
        full_train_dataset = VOCSegmentation(root=DATASET_PATH+'train', image_set="train", download=False, transforms=train_preprocess)
    sub_train_dataset = Subset(full_train_dataset, mask_dataset)

    full_val_dataset = VOCSegmentation(root=DATASET_PATH+'val', image_set="val", download=False, transforms=val_preprocess)
    sub_val_dataset = Subset(full_val_dataset, mask_dataset)

    if args.use_subset:
        train_dataset = sub_train_dataset
        val_dataset = sub_val_dataset
    else:
        train_dataset = full_train_dataset
        val_dataset = full_val_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # train model
    deeplabv3 = ResNet_DeepLabV3(num_classes=21, use_resnet101=args.use_resnet101, output_stride=args.output_stride)
    
    solver = Solver(
        deeplabv3, 
        train_loader, 
        val_loader, 
        num_classes=21,
        num_epochs=args.num_epochs, 
        lr=args.learning_rate, 
        device=device,
        log_every=args.log_every,
        verbose=args.verbose)

    solver.train()

if __name__ == '__main__':
    main()