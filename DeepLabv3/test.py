from modules.layers import *
import torch, torchvision

from torchvision.datasets import VOCSegmentation
from utils.custom_transform import *
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

torch.random.manual_seed(230)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using",device)

def check_accuracy(model, num_classes, loader):
        model.to(device)
        model.eval()

        intersect = torch.zeros(num_classes, device=device)
        union = torch.zeros(num_classes, device=device)

        with torch.no_grad():
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
        mIoU = iou.mean().cpu()
        return mIoU.item()

train_preprocess = CoCompose([
    CoToTensor,
    CoNormalize
])

val_preprocess = CoCompose([
    CoToTensor,
    CoNormalize
])

DATASET_PATH = 'dataset/'

train_dataset = VOCSegmentation(root=DATASET_PATH+'train', image_set="train", download=False)
val_dataset = VOCSegmentation(root=DATASET_PATH+'val', image_set="val", download=False)

model = torch.load('out/best_model.pth')

# plot an example
imgs, lbls = train_dataset[np.random.randint(0, 40)]

input_imgs, input_lbls = train_preprocess(imgs, lbls)
preds = model(input_imgs.unsqueeze(0).to(device))['out'].argmax(1).squeeze(0).cpu()

input_imgs = TF.pil_to_tensor(imgs)

classes = torch.zeros((21, input_imgs.shape[1], input_imgs.shape[2]))

for i in range(21):
    classes[i] = i

preds_mask = preds == classes
lbls_mask = input_lbls == classes

gt_img = torchvision.utils.draw_segmentation_masks(input_imgs, lbls_mask)
pred_img = torchvision.utils.draw_segmentation_masks(input_imgs, preds_mask)

img_grid = torchvision.utils.make_grid([gt_img, pred_img])
pil_grid = TF.to_pil_image(img_grid)
plt.imshow(pil_grid)
plt.show()





