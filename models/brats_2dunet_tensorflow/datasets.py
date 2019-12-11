import pickle
import glob
import os
import socket
import logging

import numpy as np
from math import ceil
from .nii_reader import nii_reader

logger = logging.getLogger(__name__)


def _get_dataset_func_map():
    return {
        'mnist': load_mnist,
#         'fashion-mnist': load_fashion_mnist,
#         'pubfig83': load_pubfig83,
#         'cifar10': load_cifar10,
#         'cifar20': load_cifar20,
#         'cifar100': load_cifar100,
#         'bsm': load_bsm,
#         'BraTS17': load_BraTS17,
        'BraTS17_institution': load_BraTS17_insitution,
    }


def get_dataset_list():
    return list(_get_dataset_func_map().keys())


def load_dataset(dataset, **kwargs):
    if dataset not in get_dataset_list():
        raise ValueError("Dataset {} not in list of datasets {get_dataset_list()}".format(dataset))
    return _get_dataset_func_map()[dataset](**kwargs)


def _get_dataset_dir(server=None):
    if server is None:
        server = socket.gethostname()
    server_to_path = {'spr-gpu01': os.path.join('/', 'raid', 'datasets'),
                      'spr-gpu02': os.path.join('/', 'raid', 'datasets'),
                      'edwardsb-Z270X-UD5': os.path.join('/', 'data'),
                      'msheller-ubuntu': os.path.join('/', 'home', 'msheller', 'datasets')}
    return server_to_path[server]


def _unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def _read_mnist(path, **kwargs):
    X_train, y_train = _read_mnist_kind(path, kind='train', **kwargs)
    X_test, y_test = _read_mnist_kind(path, kind='t10k', **kwargs)

    return X_train, y_train, X_test, y_test


# from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
def _read_mnist_kind(path, kind='train', one_hot=True, **kwargs):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    images = images.astype(float) / 255
    if one_hot:
        labels = _one_hot(labels.astype(np.int), 10)

    return images, labels


def load_BraTS17_insitution(institution=0, channels_first=False, **kwargs):
    path = os.path.join(_get_dataset_dir(), 'BraTS17', 'by_institution', str(institution))
    files = ['imgs_train.npy', 'msks_train.npy', 'imgs_val.npy', 'msks_val.npy']
    
    data = [np.load(os.path.join(path, f), mmap_mode='r') for f in files]
    
    if channels_first:
        data = [np.swapaxes(d, 1, 3) for d in data]
        data = [np.swapaxes(d, 2, 3) for d in data]

    return tuple(data)

def load_mnist(**kwargs):
    path = os.path.join(_get_dataset_dir(), 'mnist', 'input_data')
    return _read_mnist(path, **kwargs)


def load_fashion_mnist(**kwargs):
    path = os.path.join(_get_dataset_dir(), 'fashion-mnist')
    return _read_mnist(path, **kwargs)


def _one_hot(y, n):
    return np.eye(n)[y]


def train_val_split(features, labels, percent_train, shuffle):
    # Splits incoming feature and labels into training and validation. The value 
    # of shuffle determines whether shuffling occurs before the split is performed. 

    def split(list, idx):
        if idx < 0 or idx > len(list):
            raise ValueError("split was out of expected range.")
        return list[:idx], list[idx:]

    nb_features = len(features)
    nb_labels = len(labels)
    if nb_features!=nb_labels:
        raise RuntimeError("Number of features and labels do not match.")
    if shuffle:
        new_order = np.random.permutation(np.arange(nb_features))
        features = features[new_order]
        labels = labels[new_order]
    split_idx = int(percent_train * nb_features)
    train_features, val_features = split(list=features, idx=split_idx)
    train_labels, val_labels = split(list=labels, idx=split_idx)
    return train_features, train_labels, val_features, val_labels

def load_from_NIfTY(parent_dir, 
                    percent_train, 
                    shuffle, 
                    channels_last=True, 
                    task='whole_tumor', 
                    **kwargs):
    # Loads data from the parent directory (NIfTY files for whole brains are 
    # assumed to be contained in subdirectories of the parent directory). 
    # Performs a split of the data into training and validation, and the value 
    # of shuffle determined whether shuffling is performed before this split 
    # occurs - both split and shuffle are done in a way to 
    # keep whole brains intact. The kwargs are passed to nii_reader.

    path = os.path.join(_get_dataset_dir(), parent_dir)
    subdirs = os.listdir(path)
    subdirs.sort()
    subdir_paths = [os.path.join(path, subdir) for subdir in subdirs]
     
    imgs_all = []
    msks_all = []
    for brain_path in subdir_paths:
        these_imgs, these_msks = \
            nii_reader(brain_path=brain_path,
                       task=task, 
                       channels_last=channels_last, 
                       **kwargs)
        # the needed files where not present if a tuple of None is returned
        if these_imgs is None:
            logger.debug('Brain subdirectory: {} did not contain the needed files.'.format(brain_path))
        else:
            imgs_all.append(these_imgs)
            msks_all.append(these_msks)
    
    # converting to arrays to allow for numpy indexing used during split
    imgs_all = np.array(imgs_all)
    msks_all = np.array(msks_all)

    # note here that each is a list of 155 slices per brain, and so the 
    # split keeps brains intact
    imgs_all_train, msks_all_train, imgs_all_val, msks_all_val = \
        train_val_split(features=imgs_all, 
                        labels=msks_all, 
                        percent_train=percent_train, 
                        shuffle=shuffle)
    # now concatenate the lists
    imgs_train = np.concatenate(imgs_all_train, axis=0)
    msks_train = np.concatenate(msks_all_train, axis=0)
    imgs_val = np.concatenate(imgs_all_val, axis=0)
    msks_val = np.concatenate(msks_all_val, axis=0)

    return imgs_train, msks_train, imgs_val, msks_val