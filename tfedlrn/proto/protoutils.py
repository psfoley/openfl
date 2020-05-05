# Copyright (C) 2020 Intel Corporation

import numpy as np
from tfedlrn.proto.collaborator_aggregator_interface_pb2 import \
    ModelProto, TensorProto, ModelHeader, MetaDataProto, MetaDatumProto

def tensor_proto_to_float32_array(tensor_proto):
    # NOTE: The assumption here is that the bytes in the proto describe a float32 array
        return np.frombuffer(tensor_proto.npbytes, dtype=np.float32).reshape(tensor_proto.shape)

def model_proto_to_float32_tensor_dict(model_proto):
    tensor_dict = {}
    for tensor_proto in model_proto.tensors:
        tensor_dict[tensor_proto.name] = tensor_proto_to_float32_array(tensor_proto)
    return tensor_dict

def construct_model_proto(tensor_dict, model_id, model_version, stage_metadata=[]):
    # currently stage_metadata is assumed to be a list of metadata protos, 
    # each entry in this list a list of 'metadatum' objects,
    # each of which has a name field and a map<int32, float> field

    model_header = ModelHeader(id=model_id, version=model_version)
    
    tensor_protos = []
    for k, v in tensor_dict.items():
        tensor_protos.append(TensorProto(name=k, shape=v.shape, npbytes=v.tobytes('C')))
    
    # the stage_metadata is already assumed to be constructed as a list of metadata protos
            
    return ModelProto(header=model_header, tensors=tensor_protos, stage_metadata=stage_metadata)


def load_proto(fpath):
    with open(fpath, "rb") as f:
        loaded = f.read()
        model = ModelProto().FromString(loaded)
        return model

def dump_proto(model_proto, fpath):
    s = model_proto.SerializeToString()
    with open(fpath, "wb") as f:
        f.write(s)


def import_weights(fpath):
    """Import weight dictionary from a pbuf file."""
    
    model_proto = load_proto(fpath)
    
    return model_proto_to_float32_tensor_dict(model_proto)