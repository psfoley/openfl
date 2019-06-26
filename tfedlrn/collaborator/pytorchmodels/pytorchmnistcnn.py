import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

from ...datasets import load_dataset
from .pytorchflutils import pt_get_tensor_dict, pt_set_tensor_dict, pt_validate, pt_train_epoch, pt_create_loader


class PyTorchMNISTCNN(nn.Module):

    def __init__(self, device, train_loader=None, val_loader=None):
        super(PyTorchMNISTCNN, self).__init__()

        self.device = device
        self.init_data_pipeline(train_loader, val_loader)
        self.init_network(device)
        self.init_optimizer()

    def get_tensor_dict(self):
        return pt_get_tensor_dict(self, self.optimizer)

    def set_tensor_dict(self, tensor_dict):
        pt_set_tensor_dict(self, tensor_dict)

    def init_data_pipeline(self, train_loader, val_loader):
        if train_loader is None or val_loader is None:
            X_train, y_train, X_val, y_val = load_dataset('mnist')
            X_train = X_train.reshape([-1, 1, 28, 28])
            X_val = X_val.reshape([-1, 1, 28, 28])

        if train_loader is None:
            self.train_loader = pt_create_loader(X_train, y_train, batch_size=64, shuffle=True)
        else:
            self.train_loader = train_loader

        if val_loader is None:
            self.val_loader = pt_create_loader(X_val, y_val, batch_size=64, shuffle=True)
        else:
            self.val_loader = val_loader

    def init_network(self, device):
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def init_optimizer(self):
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

    def get_optimizer(self):
        return self.optimizer

    def train_epoch(self, epoch=None):

        def loss_fn(output, target):
            return F.cross_entropy(output, torch.max(target, 1)[1])

        return pt_train_epoch(self, self.train_loader, self.device, self.optimizer, loss_fn)
        # # set to "training" mode
        # self.train()
        
        # losses = []

        # for data, target in self.train_loader:
        #     data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
        #     self.optimizer.zero_grad()
        #     output = self(data)
        #     loss = F.cross_entropy(output, torch.max(target, 1)[1])
        #     loss.backward()
        #     self.optimizer.step()
        #     losses.append(loss.detach().cpu().numpy())

        # return np.mean(losses)

    def get_training_data_size(self):
        return len(self.train_loader.dataset)

    def validate(self):
        self.eval()
        correct = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
                output = self(data)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                target = torch.max(target, 1)[1]
                # FIXME: there has to be a better way than exhaustive eq then diagonal
                eq = pred.eq(target).diag().sum().cpu().numpy()
                correct += eq

        return correct / self.get_validation_data_size()

    def get_validation_data_size(self):
        return len(self.val_loader.dataset)
