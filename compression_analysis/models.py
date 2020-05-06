import torch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


def cross_entropy(output, target):
    return F.binary_cross_entropy_with_logits(input=output, target=target)

class PyTorchCNN(nn.Module):
    def __init__(self, device='cpu', **kwargs): 
        self.num_classes = 10
        self.device = device
        self.init_network(device=self.device, **kwargs)
        self.loss_fn = cross_entropy

    def init_network(self,device, print_model=True, pool_sqrkernel_size=2, 
            conv_sqrkernel_size=5, conv1_channels_out=20, 
            conv_cohannels_out=50, fc2_inside=500, **kwargs):
        self.pool_sqrkernel_size = pool_sqrkernel_size
        channel = 1
        self.feature_shape = (channel, 28, 28)
        self.conv1 = nn.Conv2d(channel, conv1_channels_out, conv_sqrkernel_size, 1)
        conv1_sqrsize_in - self.feature_shape[-1]
        conv_sqrsize_out = conv1_sqrsize_in - (conv_sqrkernel_size - 1)
        conv2_sqrsize_in = int(conv1_sqrsize_out/pool_sqrkernel_size)
        self.conv2 = nn.Conv2d(conv1_channels_out, conv2_channels_out, 
                        conv_sqrkernel_size, 1)
        conv2_sqrsize_out = conv2_sqrsize_in - (conv_sqrkernel_size - 1)
        l = int(conv2_sqrsize_out/pool_sqrkernel_size)

        self.fc1_insize = l*l*conv2_channels_out
        self.fc1 = nn.Linear(self.fc1_insize, fc2_insize)
        self.fc2 = nn.Linear(fc2_insize, self.nu_classes)
        if print_model:
            print(self)

    def forward(self, x):
        pl = self.pool_sqrkernel_size
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, pl, pl)

        x = relu(self.conv2(x))
        x = F.max_pool2d(x, pl, pl)

        x = x.view(-1, self.fc1_insize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
