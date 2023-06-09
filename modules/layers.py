import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from collections import OrderedDict
from typing import List

class ResNet_DeepLabV3(nn.Module):
    """DeepLabV3 model with RestNet as backbone

    Arguments:
    - num_classes: the number of classes
    - use_resnet101: use resnet101 as backbone, if False, use restnet50. Default: False
    - output_stride: the output stride of model (only support 16 or 8). Default: 16
    """
    def __init__(self, num_classes: int, use_resnet101=False, output_stride=16) -> None:
        super().__init__()

        assert output_stride in [8, 16], 'output stride must be 8 or 16'

        if use_resnet101:
            origin_resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        else:
            origin_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        m_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        m_dict = OrderedDict()
        
        train_block = ['layer4']
        for name, module in origin_resnet.named_children():
            if name in m_list:
                if name not in train_block:
                    for param in module.parameters():
                        param.requires_grad = False
                
                m_dict[name] = module

        last_blocks = ['layer4'] if output_stride == 16 else ['layer3', 'layer4']

        for i, block_name in enumerate(last_blocks):
            block = m_dict[block_name]
            m_dict[block_name] = self.make_block(block, dilation_rate=2**(i+1))

        self.backbone = nn.Sequential(m_dict)

        self.classifier = DeepLabHead(2048, num_classes)

    def forward(self, X):
        result = OrderedDict()
        input_shape = X.shape[-2:]
        features = self.backbone(X)
        logits = self.classifier(features)
        scores = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)

        result['out'] = scores
        return result
    
    def make_block(self, block, dilation_rate):
        remove_stride = ['0.conv2', '0.downsample.0']

        for name, module in block.named_modules():
            if name in remove_stride:
                module.stride = (1,1)
            
            if name == '0.conv2':
                if dilation_rate == 2: # first block of last_blocks, don't add dilation
                    continue
                else:
                    module.padding = (dilation_rate//2, dilation_rate//2)
                    module.dilation = (dilation_rate//2, dilation_rate//2)
                    continue
            
            if 'conv2' in name:
                module.padding = (dilation_rate, dilation_rate)
                module.dilation = (dilation_rate, dilation_rate)

        return block
    
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        size = X.shape[-2:]
        for mod in self:
            X = mod(X)
        
        return F.interpolate(X, size=size, mode='bilinear', align_corners=False)
    
class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates: List[int]) -> None:
        super().__init__()

        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        )

        for dilation in rates:
            modules.append(
                ASPPConv(in_channels, out_channels, dilation=dilation)
            )

        modules.append(
            ASPPPooling(in_channels, out_channels)
        )

        self.pools = nn.ModuleList(modules)

        self.proj = nn.Sequential(
            nn.Conv2d(len(self.pools) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _res = []

        for pool in self.pools:
            _res.append(pool(X))
        
        res = torch.concat(_res, dim=1)
        return self.proj(res)