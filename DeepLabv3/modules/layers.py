import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict
from typing import List

class ResNet50_DeepLabV3_16(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        origin_resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        m_list = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        m_dict = OrderedDict()
        
        for name, module in origin_resnet50.named_children():
            if name in m_list:
                if name != 'layer4':
                    for param in module.parameters():
                        param.requires_grad = False

                # for param in module.parameters():
                #     param.requires_grad = False
                
                m_dict[name] = module


        last_block = m_dict['layer4']

        for name, module in last_block.named_modules():
            remove_stride = ['0.conv2', '0.downsample.0']
            use_dilation = ['1.conv2', '2.conv2']
            if name in remove_stride:
                module.stride = (1,1)

            if name in use_dilation:
                module.padding = (2,2)
                module.dilation = (2,2)

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