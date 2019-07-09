#!/usr/bin/env python3
import argparse

import tfedlrn
from tfedlrn.gpuutils import pick_cuda_device
from tfedlrn.collaborator.tensorflowmodels.tensorflow2dunet import TensorFlow2DUNet
from tfedlrn.proto.message_pb2 import *
import os

def main(model='TensorFlow2DUNet'):
    # pick any available gpu
    pick_cuda_device()

    import tensorflow as tf

    net = globals()[model](None, None, None, None)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fname = "{}/../initial_models/{}.pbuf".format(script_dir, model)

    d = net.get_tensor_dict()

    tensor_protos = []
    for k, v in d.items():
        tensor_protos.append(TensorProto(name=k, shape=v.shape, values=v.flatten(order='C')))

    model_header = ModelHeader(id=model, version=0)

    model_proto = ModelProto(header=model_header, tensors=tensor_protos)

    s = model_proto.SerializeToString()

    with open(fname, "wb") as f:
        f.write(s)

    with open(fname, "rb") as f:
        loaded = f.read()

    loaded = ModelProto().FromString(loaded)

    assert loaded == model_proto
    
    print('created', fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, choices=['TensorFlow2DUNet'], required=True)
    args = parser.parse_args()
    main(**vars(args))
