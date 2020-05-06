import time
import os
import torch
import argparse
from logging_utils import logger
import logging
from models import PyTorchCNN



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compression_method', type=str, default='stc')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    config = parser.parse_args()
    
    #trainer = Trainer(config, dataset_train, dataset_test)
    logger.warning('compression method:: %s, dataset:: %s', config.compression_method, config.dataset)

    net = PyTorchCNN
    print(net)
    

if __name__=="__main__":
    main()
