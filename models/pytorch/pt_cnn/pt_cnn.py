# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


"""
You may copy this file as the starting point of your own model.
"""
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.pytorch import PyTorchFLModel

def cross_entropy(output, target):
    return F.binary_cross_entropy_with_logits(input=output, target=target)

        

class PyTorchCNN(PyTorchFLModel):
    """
    Simple CNN for classification.
    """

    def __init__(self, data, device='cpu', **kwargs):
        super().__init__(data=data, device=device, **kwargs)

        self.num_classes = self.data.num_classes
        self.init_network(device=self.device, **kwargs)
        self._init_optimizer()        
        self.loss_fn = cross_entropy

    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def init_network(self, 
                     device,
                     print_model=True, 
                     pool_sqrkernel_size=2,
                     conv_sqrkernel_size=5, 
                     conv1_channels_out=20, 
                     conv2_channels_out=50, 
                     fc2_insize = 500, 
                     **kwargs):
        """
        FIXME: We are tracking only side lengths (rather than length and width) as we are assuming square 
        shapes for feature and kernels.
        In order that all of the input and activation components are used (not cut off), we rely on a criterion:
        appropriate integers are divisible so that all casting to int perfomed below does no rounding 
        (i.e. all int casting simply converts a float with '0' in the decimal part to an int.)

        (Note this criterion held for the original input sizes considered for this model: 28x28 and 32x32 
        when used with the default values above)
                     
        """
        self.pool_sqrkernel_size = pool_sqrkernel_size
        channel = self.data.get_feature_shape()[0]# (channel, dim1, dim2)
        self.conv1 = nn.Conv2d(channel, conv1_channels_out, conv_sqrkernel_size, 1)

        # perform some calculations to track the size of the single channel activations
        # channels are first for pytorch
        conv1_sqrsize_in = self.feature_shape[-1]
        conv1_sqrsize_out = conv1_sqrsize_in - (conv_sqrkernel_size - 1)
        # a pool operation happens after conv1 out 
        # (note dependence on 'forward' function below)
        conv2_sqrsize_in = int(conv1_sqrsize_out/pool_sqrkernel_size)
        
        self.conv2 = nn.Conv2d(conv1_channels_out, conv2_channels_out, conv_sqrkernel_size, 1)
        
        # more tracking of single channel activation size
        conv2_sqrsize_out = conv2_sqrsize_in - (conv_sqrkernel_size - 1)
        # a pool operation happens after conv2 out
        # (note dependence on 'forward' function below)
        l = int(conv2_sqrsize_out/pool_sqrkernel_size)
        self.fc1_insize = l*l*conv2_channels_out
        self.fc1 = nn.Linear(self.fc1_insize, fc2_insize)
        self.fc2 = nn.Linear(fc2_insize, self.num_classes)
        if print_model:
            print(self)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        pl = self.pool_sqrkernel_size
        x = F.max_pool2d(x, pl, pl)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, pl, pl)
        x = x.view(-1, self.fc1_insize)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def validate(self, use_tqdm=False):
        self.eval()
        val_score = 0
        total_samples = 0

        loader = self.data.get_val_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="validate")

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

    def train_epoch(self, use_tqdm=False): 
        # set to "training" mode
        self.train()
        
        losses = []

        loader = self.data.get_train_loader()
        if use_tqdm:
            loader = tqdm.tqdm(loader, desc="train epoch")

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss_fn(output=output, target=target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach().cpu().numpy())
        # DEBUG
        print(np.mean(losses))

        return np.mean(losses)

    def reset_opt_vars(self):
        self._init_optimizer()
