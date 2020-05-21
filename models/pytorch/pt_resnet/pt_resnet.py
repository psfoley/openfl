# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


"""
You may copy this file as the starting point of your own model.
Based on the implementation here:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.pytorch import PyTorchFLModel
import torch.optim as optim

from models.pytorch import PyTorchFLModel

def cross_entropy(output, target):                                                                                                                          
    return F.binary_cross_entropy_with_logits(input=output, target=target)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class PyTorchResnet(PyTorchFLModel):

    def __init__(self, data, device='cpu', **kwargs):
        super().__init__(data=data, device=device, **kwargs)
        
        self.num_classes = self.data.num_classes
        self.init_network(self.device, BasicBlock, [2,2,2,2], **kwargs)# Resnet 18
        self._init_optimizer()
        self.loss_fn = cross_entropy
    
    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        #self.optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def init_network(self, device, block, num_blocks, num_classes=10, **kwargs):
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64 #width_per_group
        channel = self.data.get_feature_shape()[0]
        self.conv1 = nn.Conv2d(channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.to(device)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def validate(self):
        self.eval()
        val_score = 0
        total_samples = 0
        
        loader = self.data.get_val_loader()
        with torch.no_grad():
            for data, target in loader:
                samples = target.shape[0]
                total_samples += samples
                data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
                output = self(data)                
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                target_categorical = target.argmax(dim=1, keepdim=True)
                val_score += pred.eq(target_categorical).sum().cpu().numpy()

        return val_score / total_samples

    def train_epoch(self): 
        # set to "training" mode
        self.train()
        losses = []
        loader = self.data.get_train_loader()
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())

        return np.mean(losses)

    def reset_opt_vars(self):
        self._init_optimizer()

