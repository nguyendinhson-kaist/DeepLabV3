
import numpy as np
import argparse
import torchvision
from torch.utils.data import Subset
import torchvision.transforms.functional as TF
import torch

from dataset import *
from modules.layers import *
from utils.custom_transform import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str, 
        help='your checkpoint (.pth) file. It should be: "out/.../*.pth"')
    
    parser.add_argument('output_stride', type=int, choices=[8,16],
        help='output stride of model')
    parser.add_argument('--use-resnet101', action='store_true',
        help='use resnet101 instead of resnet50')

    parser.add_argument('--gpu-id', type=int, default=0,
        help='select gpu if there are many gpus, be sure id is valid')
    parser.add_argument('--num-samples', type=int, default=1,
        help='the number of samples you want to predict')

    args = parser.parse_args()
    return args

def predict():
    args = parse_args()

    # select device
    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Using",device)

    # load model
    model = ResNet_DeepLabV3(num_classes=21, use_resnet101=args.use_resnet101, output_stride=args.output_stride)
    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    model.eval()
    # print(model)

    # load dataset
    preprocess = CoCompose([
        CoToTensor,
        CoNormalize
    ])

    DATASET_PATH = 'dataset/'
    val_dataset = VOCSegmentation(root=DATASET_PATH+'val', image_set="val", download=False)

    samples_indexes = np.random.permutation(range(val_dataset.__len__()))[:args.num_samples]
    
    subset = Subset(val_dataset, samples_indexes)
    results = []

    for i in range(args.num_samples):
        img, gt = subset[i]

        input_img, input_gt = preprocess(img, gt)

        with torch.no_grad():
            input_img = input_img.to(device=device, dtype=torch.float32)

            pred = model(input_img.unsqueeze(0).to(device))['out'].detach().argmax(1).squeeze(0).cpu()

            results.append((img, input_gt, pred))

    # print out all results
    cmap = voc_cmap()
    segmented_cmap = [tuple(c) for c in cmap.tolist()]
    fig, axs = plt.subplots(nrows=args.num_samples, ncols=4)
    for i in range(args.num_samples):
        img, input_gt, pred = results[i]
        segmented_img = create_segmented_img(img, pred, segmented_cmap)

        input_gt = cmap[input_gt].astype('uint8')
        pred = cmap[pred].astype('uint8')
        axs[i, 0].imshow(img)
        axs[i, 0].axis('off')

        axs[i, 1].imshow(input_gt)
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(pred)
        axs[i, 2].axis('off')

        axs[i, 3].imshow(segmented_img)
        axs[i, 3].axis('off')

    plt.show()

def create_segmented_img(img, mask, cmap, num_classes=21):
    """Create pil image with mask

    Arguments:
    - img: Pil input image
    - mask: segmentation mask (tensor)

    Output:
    - segmented images (pil)
    """
    input_img = TF.pil_to_tensor(img)

    segmented_img = torchvision.utils.draw_segmentation_masks(
        input_img, 
        mask == torch.arange(num_classes)[:, None, None],
        alpha=0.5,
        colors=cmap
    )

    return TF.to_pil_image(segmented_img)

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


if __name__ == '__main__':
    predict()