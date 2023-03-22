from modules.layers import *
import torch, torchvision

from torch.utils.data import DataLoader
from utils.custom_transform import *
from dataset import VOCSegmentation
import torchvision.transforms.functional as TF

from utils.metric import SemanticMetric

import argparse

torch.random.manual_seed(230)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str, 
        help='your checkpoint (.pth) file. It should be: "out/.../*.pth"')
    parser.add_argument('output_stride', type=int, choices=[8,16],
        help='output stride of model')
    
    parser.add_argument('--gpu-id', type=int, default=0,
        help='select gpu if there are many gpus, be sure id is valid')

    args = parser.parse_args()
    return args

def check_accuracy(model, num_classes, loader, device):
        metric = SemanticMetric(num_classes=num_classes)
        model.to(device)
        model.eval()

        with torch.no_grad():
            for X, y in loader:
                X = X.to(device=device, dtype=torch.float32)
                y = y.detach().numpy().astype(int)

                pred = model(X)['out'].detach().argmax(1).cpu().numpy().astype(int)

                # update metric
                metric.update(y.flatten(), pred.flatten())

            mIoU, iou = metric.get_results()

        return mIoU, iou.tolist()

def test():
    args = parse_args()

    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Using",device)

    # load model
    model = ResNet50_DeepLabV3(num_classes=21, output_stride=args.output_stride)
    model.load_state_dict(torch.load(args.checkpoint))
    # model = torchvision.models.segmentation.deeplabv3_resnet50(weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights)

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
