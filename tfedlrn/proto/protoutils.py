import numpy as np
from tfedlrn.proto.message_pb2 import ModelProto

def repeated_values_to_nparray(repeated_values):
    return np.array(list(repeated_values))

def tensor_proto_to_nparray(tensor, reshape=True):
    array = repeated_values_to_nparray(tensor.values)
    if reshape:
        array = array.reshape(tensor.shape)
    return array

def load_proto(fpath):
    with open(fpath, "rb") as f:
        loaded = f.read()
        model = ModelProto().FromString(loaded)
        return model

def dump_proto(model_proto, fpath):
    s = model_proto.SerializeToString()

    with open(fpath, "wb") as f:
        f.write(s)