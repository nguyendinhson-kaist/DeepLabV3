import torchvision.transforms.functional as TF
import torch
import random
import numpy as np
import torchvision.transforms as transforms

class CoRandomResized:
    """Random resize the image with a given random scale range

    Args:
        scale (tuple): a tuple of (upper, lower) range value
    """
    def __init__(self, scale):
        assert isinstance(scale, tuple)
        self.scale = scale

    def __call__(self, img, target):
        w, h = img.size

        # pick randomly a scale
        s = random.uniform(*self.scale)

        new_w, new_h = int(w*s), int(h*s)

        new_img = TF.resize(img, (new_h, new_w))
        new_target = TF.resize(target, (new_h, new_w))

        return new_img, new_target
    
def CoRandomCrop(img, target):
    """Crop image with random location"""

    o_size = 513
    w, h = img.size

    # random starting location
    s_h = random.randint(1, h) - int(o_size/2)
    s_w = random.randint(1, w) - int(o_size/2)

    new_img = TF.crop(img, s_h, s_w, o_size, o_size)
    new_target = TF.crop(target, s_h, s_w, o_size, o_size)

    return new_img, new_target

def CoCenterCrop(img, target):
    """Crop image with random location"""

    o_size = 513
    w, h = img.size

    # random starting location
    s_h = int((h-o_size)/2)
    s_w = int((w-o_size)/2)

    new_img = TF.crop(img, s_h, s_w, o_size, o_size)
    new_target = TF.crop(target, s_h, s_w, o_size, o_size)

    return new_img, new_target

def CoRandomFlip(img, target):
    """Randomly flip image"""

    new_img = img
    new_target = target

    hflip = random.random() < 0.5
    if hflip:
        new_img = TF.hflip(new_img)
        new_target = TF.hflip(new_target)

    # vflip = random.random() < 0.5
    # if vflip:
    #     new_img = TF.vflip(new_target)
    #     new_target = TF.vflip(new_target)

    return new_img, new_target

def CoToTensor(img, target):
    """Convert to tensor"""
    new_img = TF.to_tensor(img)
    new_target = torch.as_tensor(np.array(target), dtype=torch.int64)
    return new_img, new_target

def CoNormalize(img, target):
    """Normalize img using mean and std of imagenet dataset"""
    new_img = TF.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    new_target = target
    return new_img, new_target
    
class CoCompose:
    """Composer for transform in both image and target

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target