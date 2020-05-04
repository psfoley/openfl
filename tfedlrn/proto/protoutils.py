# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np
from tfedlrn.proto.collaborator_aggregator_interface_pb2 import ModelProto, TensorProto, ModelHeader

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


def export_weights(model_name, version, tensor_dict, fpath):
    """
    Export the model weights to serialized protobuf.

    Parameters
    ----------
    model_name : str
        The model name in header
    version : int
        The model version in header (initial 0)
    tensor_dict : dict
        The weights dictionary.
    fpath : str
        The file path to export.
    """

    d = tensor_dict

    tensor_protos = []
    for k, v in d.items():
        tensor_protos.append(TensorProto(name=k, shape=v.shape, npbytes=v.tobytes('C')))

    model_header = ModelHeader(id=model_name, version=version)
    model_proto = ModelProto(header=model_header, tensors=tensor_protos)
    
    dump_proto(model_proto, fpath)
    print('created', fpath)

def import_weights(fpath):
    """Import weight dictionary from a pbuf file."""
    
    model_proto = load_proto(fpath)
    
    tensor_dict = {}
    for tensor_proto in model_proto.tensors:
        # FIXME: Not all parameters are numpy arrays, and may not always have data type float32
        # currently this breaks if an Adam optimizer is used as one parameter is an int.
        tensor_dict[tensor_proto.name] = np.frombuffer(tensor_proto.npbytes, dtype=np.float32).reshape(tensor_proto.shape)

    return tensor_dict