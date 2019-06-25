#!/usr/bin/env python3
import argparse

import tensorflow as tf

import tfedlrn
from tfedlrn.collaborator.tensorflowmodels.tensorflow2dunet import TensorFlow2DUNet
from tfedlrn.proto.message_pb2 import *


def main(model='TensorFlow2DUNet'):
    net = globals()[model](None, None, None, None)
    fname = "../initial_models/{}.pbuf".format(model)

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
