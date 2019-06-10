#!/usr/bin/env python3
import argparse
import numpy as np

import tfedlrn
import tfedlrn.collaborator
from tfedlrn.collaborator.collaborator import Collaborator
from tfedlrn.zmqconnection import ZMQClient

from tfedlrn.collaborator.pytorchmodels.pytorchmnistcnn import PyTorchMNISTCNN
from tfedlrn.collaborator.pytorchmodels.pytorch2dunet import PyTorch2DUNet


from tfedlrn.datasets import load_dataset

from tfedlrn.proto.message_pb2 import *

import torch


def create_loader(X, y, **kwargs):
    tX = torch.stack([torch.Tensor(i) for i in X])
    ty = torch.stack([torch.Tensor(i) for i in y])
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tX, ty), **kwargs)


def main(col_num=0, num_collaborators=4, model_id='PyTorchMNISTCNN', device='cuda'):
    agg_id = "simple pytorch agg"
    fed_id = "simple pytorch fed"
    col_id = "simple pytorch col {}".format(col_num)

    connection = ZMQClient('{} connection'.format(col_id))

    # load our data
    if model_id == 'PyTorchMNISTCNN':
        X_train, y_train, X_val, y_val = load_dataset('mnist')

        X_train = X_train[col_num::num_collaborators]
        y_train = y_train[col_num::num_collaborators]
        X_val = X_val[col_num::num_collaborators]
        y_val = y_val[col_num::num_collaborators]

        X_train = X_train.reshape([-1, 1, 28, 28])
        X_val = X_val.reshape([-1, 1, 28, 28])
    elif model_id == 'PyTorch2DUNet':
        X_train, y_train, X_val, y_val = load_dataset('BraTS17_institution',
                                                      institution=col_num,
                                                      channels_first=True)
    else:
        raise NotImplementedError('No model_id {}'.format(model_id))

    train_loader = create_loader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = create_loader(X_val, y_val, batch_size=64, shuffle=True)

    device = torch.device(device)

    model = globals()[model_id](device, train_loader=train_loader, val_loader=val_loader)

    col = Collaborator(col_id, agg_id, fed_id, model, connection, model_id, -1)

    col.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--col_num', '-c', type=int, default=0)
    parser.add_argument('--num_collaborators', '-n', type=int, default=4)
    parser.add_argument('--model_id', '-m', type=str, choices=['PyTorchMNISTCNN', 'PyTorch2DUNet'], required=True)
    parser.add_argument('--device', '-d', type=str, default='cuda')
    args = parser.parse_args()
    main(**vars(args))
