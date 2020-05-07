# Copyright (C) 2020 Intel Corporation

import numpy as np
from tfedlrn.proto.collaborator_aggregator_interface_pb2 import \
    ModelProto, TensorProto, ModelHeader, MetadataProto

def tensor_proto_to_float32_array(tensor_proto):
    # NOTE: The assumption here is that the bytes in the proto describe a float32 array
        return np.frombuffer(tensor_proto.npbytes, dtype=np.float32).reshape(tensor_proto.shape)

def model_proto_to_tensors_and_metadata(model_proto):
    tensor_dict = {}
    metadata_dict = {}
    for tensor_proto in model_proto.tensors:
        tensor_dict[tensor_proto.name] = tensor_proto_to_float32_array(tensor_proto)
        metadata_dict[tensor_proto.name] = [proto.int_to_float for proto in tensor_proto.transformer_metadata]
    return tensor_dict, metadata_dict

def tensors_and_metadata_to_model_proto(tensor_dict, model_id, model_version, metadata_dict={}):
    # transformer_dict keys are a subset of the tensor_dict keys, each value is 
    # a transformer metadata list, each entry of the list providing metadata for the backward method call
    # of a particular transformer of the custom array transformation pipeline (if one is specified in the flplan). 
    # The metadata for one transformer is a dictionary with integer keys and float values.

    model_header = ModelHeader(id=model_id, version=model_version)
    
    tensor_protos = []
    for key, array in tensor_dict.items():
        metadata_protos = [MetadataProto(int_to_float=entry) for entry in metadata_dict[key]]
        tensor_protos.append(TensorProto(name=key, 
                                         shape=array.shape, 
                                         npbytes=array.tobytes('C'), 
                                         transformer_metadata=metadata_protos))
    
    return ModelProto(header=model_header, tensors=tensor_protos)


def construct_proto(tensor_dict, model_id, model_version, compression_pipeline):
    # compress the arrays in the tensor_dict, and collect metadata for decompression
    # TODO: Hold-out tensors from the compression pipeline.
    compressed_tensor_dict = {}
    transformer_metadata_dict = {}
    for key, array in tensor_dict.items():
        compressed_tensor_dict[key], transformer_metadata_dict[key] = compression_pipeline.forward(data=array)
    
    # convert the compressed_tensor_dict and metadata to protobuf, and make the new model proto
    model_proto = tensors_and_metadata_to_model_proto(tensor_dict= compressed_tensor_dict, 
                                                      model_id=model_id, 
                                                      model_version=model_version, 
                                                      metadata_dict=transformer_metadata_dict)
    return model_proto


def deconstruct_proto(model_proto, compression_pipeline):
    # extract the tensor_dict and metadata
    tensor_dict, metadata_dict = model_proto_to_tensors_and_metadata(model_proto)
            
    # decompress the tensors
    # TODO: Handle tensors meant to be held-out from the compression pipeline (currently none are held out).
    tensor_dict = {key: compression_pipeline.backward(data=tensor_dict[key], 
                                                      transformer_metadata=metadata_dict[key])
                   for key in tensor_dict}
    return tensor_dict

def load_proto(fpath):
    with open(fpath, "rb") as f:
        loaded = f.read()
        model = ModelProto().FromString(loaded)
        return model

def dump_proto(model_proto, fpath):
    s = model_proto.SerializeToString()
    with open(fpath, "wb") as f:
        f.write(s)