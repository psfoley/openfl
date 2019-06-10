#!/usr/bin/env python3
import argparse
import os

import numpy as np

import tfedlrn
import tfedlrn.aggregator
from tfedlrn.aggregator.aggregator import Aggregator
from tfedlrn.zmqconnection import ZMQServer

from tfedlrn.proto.message_pb2 import *


def main(num_collaborators=4, initial_model='PyTorchMNISTCNN'):
    agg_id = "simple pytorch agg"
    fed_id = "simple pytorch fed"
    col_ids = ["simple pytorch col {}".format(i) for i in range(num_collaborators)]

    connection = ZMQServer('{} connection'.format(agg_id))

    with open(os.path.join('..', 'initial_models', "{}.pbuf".format(initial_model)), "rb") as f:
        loaded = f.read()
    model = ModelProto().FromString(loaded)

    agg = Aggregator(agg_id, fed_id, col_ids, connection, model)

    agg.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_collaborators', '-n', type=int, default=4)
    parser.add_argument('--initial_model', '-i', type=str, default='PyTorchMNISTCNN')
    args = parser.parse_args()
    main(**vars(args))
