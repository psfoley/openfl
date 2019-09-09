from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

from ...datasets import brats17_data_paths, get_data_reader
from .pytorchflutils import pt_get_tensor_dict, pt_set_tensor_dict, pt_validate, \
    pt_train_epoch, pt_create_pipeline_loader
from .pytorchflutils_test import pt_train_epoch_test_performance

# FIXME: Put docstrings into all functions


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


class PyTorch2DUNetDPipe(nn.Module):
    # Similar to PyTorch2DUNet, but with data pipeline that loads single 2D images
    # from disk using directories in NIfTI format. Since this is a 2D UNet model
    # and our NIfTI directories contain

    def __init__(self, device, train_loader=None, val_loader=None, 
      optimizer='SGD', dropout_layers=[2, 3], shuffle=True):
        super(PyTorch2DUNetDPipe, self).__init__()

        if dropout_layers is None:
            self.dropout_layers = []
        else:
            self.dropout_layers = dropout_layers

        # including these attributes for testing
        self.read_train = None
        self.idx_to_train_paths = None

        self.device = device
        self.init_data_pipeline(train_loader, val_loader, shuffle)
        self.init_network(device)
        self.init_optimizer(optimizer)



    def get_tensor_dict(self):
        return pt_get_tensor_dict(self, self.optimizer)

    def set_tensor_dict(self, tensor_dict):
        pt_set_tensor_dict(self, tensor_dict)

    @staticmethod
    def _tensorize_data(X, y):
        tX = torch.stack([torch.Tensor(i) for i in X])
        ty = torch.stack([torch.Tensor(i) for i in y])
        return torch.utils.data.TensorDataset(tX, ty)

    def init_data_pipeline(self, train_loader, val_loader, shuffle, 
      label_type='whole_tumor', **kwargs):

        if not shuffle:
            print("WARNING: SHUFFLE IS DISABLED !!!!!")

        if train_loader is None or val_loader is None:

            # train and val paths are lists of tuples: (X_path, y_path)
            # each path corresponds to a file for a single sample  
            idx_to_train_paths, train_data_length = brats17_data_paths('BraTS17_train')
            idx_to_val_paths, val_data_length = brats17_data_paths('BraTS17_val')

            # exposing these objects for testing purposes
            self.idx_to_train_paths = idx_to_train_paths
            self.read_train = get_data_reader('BraTS17_{}'.format(label_type),
              idx_to_train_paths, channels_last=False)

            read_and_preprocess_train = get_data_reader('BraTS17_{}'.format(label_type),
              idx_to_train_paths, channels_last=False)
            read_and_preprocess_val = get_data_reader('BraTS17_{}'.format(label_type),
              idx_to_val_paths, channels_last=False)

        if train_loader is None:
            self.train_loader = pt_create_pipeline_loader(read_and_preprocess_train, 
              length=train_data_length, 
              batch_size=64, shuffle=shuffle)
        else:
            self.train_loader = train_loader

        if val_loader is None:
            self.val_loader = pt_create_pipeline_loader(read_and_preprocess_val, 
                                               length=val_data_length, 
                                               batch_size=64, shuffle=shuffle)
        else:
            self.val_loader = val_loader
            
    def init_network(self,
                     device,
                     initial_channels=1,
                     depth_per_side=5,
                     initial_filters=32):

        f = initial_filters
        
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
        
        # send this to the device
        self.to(device)
        
    def forward(self, x):

        # DEBUG
        print("input x has shape: {}".format(x.size()))
        
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
            raise ValueError()

    def get_optimizer(self):
        return self.optimizer

    def train_epoch(self, epoch=None):
        # FIXME: update to proper training schedule when architected
        if epoch == 8:
            self.init_optimizer('RMSprop')

        loss_fn = partial(dice_coef_loss, smoothing=32.0)

        return pt_train_epoch(self, self.train_loader, self.device, self.optimizer, loss_fn)


    
    def train_epoch_test_performance(self, num_batches, epoch=None):
        # FIXME: update to proper training schedule when architected
        if epoch == 8:
            self.init_optimizer('RMSprop')

        loss_fn = partial(dice_coef_loss, smoothing=32.0)

        loss,  data_load_times, train_times = \
          pt_train_epoch_test_performance(self, self.train_loader, \
              self.device, self.optimizer, loss_fn,  num_batches)
        mean_train_times = np.mean(train_times)
        std_train_times = np.std(train_times)
        mean_data_load_times = np.mean(data_load_times)
        std_data_load_times = np.std(data_load_times)
        return loss, mean_train_times, std_train_times, \
            mean_data_load_times, std_data_load_times


    def get_training_data_size(self):
        return len(self.train_loader.dataset)

    def validate(self):
        return pt_validate(self, self.val_loader, self.device, dice_coef)

    def get_validation_data_size(self):
        return len(self.val_loader.dataset)
