from modules.layers import *
import torch, torchvision

from torch.utils.data import DataLoader
from utils.custom_transform import *
from dataset import VOCSegmentation
import torchvision.transforms.functional as TF

import argparse

torch.random.manual_seed(230)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str, 
        help='your checkpoint (.pth) file. It should be: "out/.../*.pth"')
    
    parser.add_argument('--gpu-id', type=int, default=0,
        help='select gpu if there are many gpus, be sure id is valid')

    args = parser.parse_args()
    return args

def check_accuracy(model, num_classes, loader, device):
        model.to(device)
        model.eval()

        with torch.no_grad():
            intersect = torch.zeros(num_classes, device=device)
            union = torch.zeros(num_classes, device=device)

            for X, y in loader:
                X = X.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.long)
                out = model(X)['out'].argmax(1)

                void_mask = y == 255
                out[void_mask] = 255

                for i_class in range(num_classes):
                    gt_mask = y == i_class
                    gt = torch.zeros_like(y)
                    gt[gt_mask] = 1.

                    out_mask = out == i_class
                    pred = torch.zeros_like(out)
                    pred[out_mask] = 1.

                    intersect_batch = (gt*pred).sum()
                    union_batch = (gt+pred).sum() - intersect_batch

                    intersect[i_class] += intersect_batch
                    union[i_class] += union_batch

            union[union == 0] = 1e-7
            iou = intersect/union
            mIoU = iou.mean()
        return mIoU.item(), iou.tolist()

def test():
    args = parse_args()

    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Using",device)

    # load model
    model = ResNet50_DeepLabV3_16(num_classes=21)
    model.load_state_dict(torch.load(args.checkpoint))

    # load val dataset
    val_preprocess = CoCompose([
        CoResize,
        CoCenterCrop,
        CoToTensor,
        CoNormalize
    ])

    DATASET_PATH = 'dataset/'
    val_dataset = VOCSegmentation(root=DATASET_PATH+'val', image_set="val", download=False, transforms=val_preprocess)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    mIoU, IoU = check_accuracy(model, num_classes=21, loader=val_loader, device=device)

    print('mIoU: %.4f' % (mIoU*100))
    print('IoU per class: ', IoU)

if __name__ == '__main__':
    test()
