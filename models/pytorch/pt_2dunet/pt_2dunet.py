# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from functools import partial
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.pytorch import PyTorchFLModel

# FIXME: move to some custom losses.py file?
def dice_coef(pred, target, smoothing=1.0):    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3))
    
    return ((2 * intersection + smoothing) / (union + smoothing)).mean()


def dice_coef_loss(pred, target, smoothing=1.0):    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target).sum(dim=(1, 2, 3))
    
    term1 = -torch.log(2 * intersection + smoothing)
    term2 = torch.log(union + smoothing)
    
    return term1.mean() + term2.mean()


class PyTorch2DUNet(PyTorchFLModel):

    def __init__(self, data, device='cpu', optimizer='SGD', **kwargs):
        super().__init__(data=data, device=device, **kwargs)

        self.init_network(device=self.device, **kwargs)
        self.init_optimizer(optimizer)
        self.loss_fn = partial(dice_coef_loss, smoothing=1.0)
   
    def train_epoch(self, epoch=None, use_tqdm=False):
        # FIXME: update to proper training schedule when architected
        if epoch == 8:
            self.init_optimizer('RMSprop')

        # set to "training" mode
        self.train()
        
        losses = []
        
        gen = self.data.get_train_loader()
        if use_tqdm:
            gen = tqdm.tqdm(gen, desc="training epoch")
        
        for data, target in gen:
            if isinstance(data, np.ndarray):
                    data = torch.Tensor(data)
            if isinstance(target, np.ndarray):
                target = torch.Tensor(data)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        return np.mean(losses)

    def validate(self, use_tqdm=False):       
        self.eval()
        val_score = 0
        total_samples = 0

        gen = self.data.get_val_loader()
        if use_tqdm:
            gen = tqdm.tqdm(gen, desc="validate")

        with torch.no_grad():
            for data, target in gen:
                samples = target.shape[0]
                total_samples += samples
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                val_score += dice_coef(output, target).cpu().numpy() * samples
        return val_score / total_samples

    def reset_opt_vars(self):
        self.init_optimizer(self.optimizer.__class__.__name__)
           
    def init_network(self,
                     device,
                     print_model=True,
                     dropout_layers=[2, 3],
                     initial_channels=1,
                     depth_per_side=5,
                     initial_filters=32, 
                     **kwargs):

        f = initial_filters
        if dropout_layers is None:
            self.dropout_layers = []
        else:
            self.dropout_layers = dropout_layers

        # store our depth for our forward function
        self.depth_per_side = 5
        
        # parameter-less layers
        self.dropout = nn.Dropout(p=0.2)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        # initial down layers
        conv_down_a = [nn.Conv2d(initial_channels, f, 3, padding=1)]
        conv_down_b = [nn.Conv2d(f, f, 3, padding=1)]
                
        # rest of the layers going down
        for i in range(1, depth_per_side):
            f *= 2
            conv_down_a.append(nn.Conv2d(f // 2, f, 3, padding=1))                
            conv_down_b.append(nn.Conv2d(f, f, 3, padding=1))
            
        # going up, do all but the last layer
        conv_up_a = []
        conv_up_b = []
        for _ in range(depth_per_side-1):
            f //= 2
            # triple input channels due to skip connections
            conv_up_a.append(nn.Conv2d(f*3, f, 3, padding=1))
            conv_up_b.append(nn.Conv2d(f, f, 3, padding=1))
            
        # do the last layer
        self.conv_out = nn.Conv2d(f, 1, 1, padding=0)
        
        # all up/down layers need to to become fields of this object
        for i, (a, b) in enumerate(zip(conv_down_a, conv_down_b)):
            setattr(self, 'conv_down_{}a'.format(i+1), a)
            setattr(self, 'conv_down_{}b'.format(i+1), b)
            
        # all up/down layers need to to become fields of this object
        for i, (a, b) in enumerate(zip(conv_up_a, conv_up_b)):
            setattr(self, 'conv_up_{}a'.format(i+1), a)
            setattr(self, 'conv_up_{}b'.format(i+1), b)
        
        if print_model:
            print(self)

        # send this to the device
        self.to(device)
        
    def forward(self, x):
        
        # gather up our up and down layer members for easier processing
        conv_down_a = [getattr(self, 'conv_down_{}a'.format(i+1)) for i in range(self.depth_per_side)]
        conv_down_b = [getattr(self, 'conv_down_{}b'.format(i+1)) for i in range(self.depth_per_side)]
        conv_up_a = [getattr(self, 'conv_up_{}a'.format(i+1)) for i in range(self.depth_per_side - 1)]
        conv_up_b = [getattr(self, 'conv_up_{}b'.format(i+1)) for i in range(self.depth_per_side - 1)]
        
        # we concatenate the outputs from the b layers
        concat_me = []
        pool = x

        # going down, wire each pair and then pool except the last
        for i, (a, b) in enumerate(zip(conv_down_a, conv_down_b)):
            out_down = F.relu(a(pool))
            if i in self.dropout_layers:
                out_down = self.dropout(out_down)
            out_down = F.relu(b(out_down))
            # if not the last down b layer, pool it and add it to the concat list
            if b != conv_down_b[-1]:
                concat_me.append(out_down)
                pool = self.maxpool(out_down) # feed the pool into the next layer
        
        # reverse the concat_me layers
        concat_me = concat_me[::-1]

        # we start going up with the b (not-pooled) from previous layer
        in_up = out_down

        # going up, we need to zip a, b and concat_me
        for a, b, c in zip(conv_up_a, conv_up_b, concat_me):
            up = torch.cat([self.upsample(in_up), c], dim=1)
            up = F.relu(a(up))
            in_up = F.relu(b(up))
        
        # finally, return the output
        return torch.sigmoid(self.conv_out(in_up))

    def init_optimizer(self, optimizer='SGD'):
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        elif optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=1e-5, momentum=0.9)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=1e-5)
        else:
            raise ValueError('Optimizer: {} is not curently supported'.format(optimizer))

