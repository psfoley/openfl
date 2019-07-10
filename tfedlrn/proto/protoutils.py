import numpy as np

def repeated_values_to_nparray(repeated_values):
    return np.array(list(repeated_values))

def tensor_proto_to_nparray(tensor, reshape=True):
    array = repeated_values_to_nparray(tensor.values)
    if reshape:
        array = array.reshape(tensor.shape)
    return array
