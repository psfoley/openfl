import numpy as np

from tfedlrn.proto.collaborator_aggregator_interface_pb2 import TensorProto, ModelProto, ModelHeader

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
    s = model_proto.SerializeToString()

    with open(fpath, "wb") as f:
        f.write(s)

    with open(fpath, "rb") as f:
        loaded = f.read()

    loaded = ModelProto().FromString(loaded)

    assert loaded == model_proto
    print('created', fpath)

def import_weights(fpath):
    """Import weight dictionary from a pbuf file."""
    with open(fpath, "rb") as f:
        loaded = f.read()

    model_proto = ModelProto().FromString(loaded)

    tensor_dict = {}
    for tensor_proto in model_proto.tensors:
        tensor_dict[tensor_proto.name] = np.frombuffer(tensor_proto.npbytes, dtype=np.float32).reshape(tensor_proto.shape)

    return tensor_dict