import torch 
import torchvision
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from utils.custom_transform import *
from utils.solver import Solver
from modules.layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using",device)

torch.random.manual_seed(230)

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
    CoCenterCrop,
    CoToTensor,
    CoNormalize
])

mask_dataset = range(10*4)

train_dataset = torchvision.datasets.VOCSegmentation(root=DATASET_PATH+'train', image_set="train", download=False, transforms=train_preprocess)
sub_train_dataset = Subset(train_dataset, mask_dataset)
train_loader = DataLoader(sub_train_dataset, batch_size=4, shuffle=False)

val_dataset = torchvision.datasets.VOCSegmentation(root=DATASET_PATH+'val', image_set="val", download=False, transforms=val_preprocess)
sub_val_dataset = Subset(val_dataset, mask_dataset)
val_loader = DataLoader(sub_val_dataset, batch_size=4, shuffle=False)

deeplabv3_16 = ResNet50_DeepLabV3_16(num_classes=21)

solver = Solver(
    deeplabv3_16, 
    train_loader, 
    val_loader, 
    num_classes=21,
    num_epochs=20, 
    lr=0.0005, 
    device=device)

solver.train()