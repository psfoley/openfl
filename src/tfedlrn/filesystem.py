import os
import pickle


def unpickle_file(file):
    if not os.path.exists(file):
        return None
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def pickle_file(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
