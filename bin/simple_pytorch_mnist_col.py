#!/usr/bin/env python3
import argparse
import numpy as np

import tfedlrn
import tfedlrn.collaborator
from tfedlrn.collaborator.collaborator import Collaborator
from tfedlrn.zmqconnection import ZMQClient

from tfedlrn.collaborator.pytorchmodels.pytorchflmnistcnn import PyTorchMNISTCNN


from tfedlrn.datasets import load_dataset

from tfedlrn.proto.message_pb2 import *

import torch


def create_loader(X, y, **kwargs):
    tX = torch.stack([torch.Tensor(i) for i in X])
    ty = torch.stack([torch.Tensor(i) for i in y])
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tX, ty), **kwargs)


def main(col_num=0, num_collaborators=4):
    agg_id = "notebook_agg"
    fed_id = "PT MNIST notebook fed"
    col_id = "notebook col {}".format(col_num)

    connection = ZMQClient('{} connection'.format(col_id))

    X_train, y_train, X_val, y_val = load_dataset('mnist')

    X_train = X_train[col_num::num_collaborators]
    y_train = y_train[col_num::num_collaborators]
    X_val = X_val[col_num::num_collaborators]
    y_val = y_val[col_num::num_collaborators]

    X_train = X_train.reshape([-1, 1, 28, 28])
    X_val = X_val.reshape([-1, 1, 28, 28])

    train_loader = create_loader(X_train, y_train, batch_size=64, shuffle=True)
    val_loader = create_loader(X_val, y_val, batch_size=64, shuffle=True)

    device = torch.device("cuda")

    model = PyTorchMNISTCNN(device, train_loader=train_loader, val_loader=val_loader)

    col = Collaborator(col_id, agg_id, fed_id, model, connection, 'pytorch_mnist', -1)

    col.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--col_num', '-c', type=int, default=0)
    parser.add_argument('--num_collaborators', '-n', type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))
