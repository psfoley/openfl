import pickle
import glob
import os
import socket
from functools import partial

import numpy as np
from math import ceil
from .brats17_reader import brats17_reader


# FIXME: put a logging command next to raised exception for data_type value error
# FIXME: Put docstrings into all functions



def brats17_data_paths(data_name):
    # used for setting up data loaders that load individual samlples from 
    # FIXME: currently the validation set is the same as the training set
    # FIXME: add support for other image datasets?
    # FIXME: add for tensorflow, even though graph pipelines work there?

    # produce a dictionary of indices to file_paths
    if data_name == 'BraTS17_train':
        directory = os.path.join(_get_dataset_dir(), 
                                 'BraTS17/MICCAI_BraTS17_Data_Training/HGG')
        paths = [os.path.join(directory, subdir) for subdir in os.listdir(directory)]
    elif data_name == 'BraTS17_val':
        directory = os.path.join(_get_dataset_dir(), 
                                 'BraTS17/MICCAI_BraTS17_Data_Training/HGG')
        paths = [os.path.join(directory, subdir) for subdir in os.listdir(directory)]
    # FIXME: temporarily disabling below
    # order paths so as to deterministically determine indices
    # paths.sort()
    nb_imgs = 155 * len(paths) 

    idx_to_paths = {idx: paths[idx // 155] for idx in range(nb_imgs)}

    return idx_to_paths, nb_imgs

def get_data_reader(data_type, idx_to_paths, channels_last):
    if data_type.startswith('BraTS17_'):
        label_type = data_type[8:]
        return partial(brats17_reader, idx_to_paths=idx_to_paths, 
                       label_type=label_type, 
                       channels_last=channels_last)
    else:
        raise ValueError("The data_type:{} is not supported.".format(data_type))


def get_data_paths(data_name):
    # FIXME: Currently the training and validation data is the same.
    if data_name.startswith('BraTS17_'):
        return brats17_data_paths(data_name)
    else:
        raise ValueError("The data_name:{} is not supported for pipelining.".format(data_name))
    


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


def load_from_NIfTY(parent_dir, channels_last=True, label_type='whole_tumor', **kwargs):
    path = os.path.join(_get_dataset_dir(), parent_dir)
    subdirs = os.listdir()
    nb_brains = len(subdirs)
    # using a tuple for the idx value indicates to brats17_reader that
    # we want to pull out full brains rather than a single slice.
    # In this case the first component of the tuple is all that is
    # needed to be present in the keys to idx_to_paths
    idxs = zip(155*np.arange(nb_brains), 155*(np.arange(nb_brains)+1))
    idx_to_paths = {idx[0]: subdirs[int(idx[0]/155)] for idx in idxs}
    imgs_all = []
    msks_all = []
    for idx in idxs:
        these_imgs, these_msks = brats17_reader(idx, idx_to_paths, 
          channels_last=channels_last, label_type=label_type, **kwargs)
        imgs_all.append(these_imgs)
        msks_all.append(these_msks)
    imgs_all = np.concatenate(imgs_all)
    msks_all = np.concatenate(msks_all)
    imgs_train, imgs_val = train_val_split(imgs_all)
    msks_train, msks_val = train_val_split(msks_all)
    return imgs_train, msks_train, imgs_val, msks_val


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