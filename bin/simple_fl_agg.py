#!/usr/bin/env python3
import argparse
import os

import numpy as np

import tfedlrn
import tfedlrn.aggregator
from tfedlrn.aggregator.aggregator import Aggregator
from tfedlrn.zmqconnection import ZMQServer

from tfedlrn.proto.message_pb2 import *


def main(num_collaborators=4, initial_model='PyTorchMNISTCNN', server_port=5555):
    agg_id = "simple agg"
    fed_id = "simple fed"
    col_ids = ["simple col {}".format(i) for i in range(num_collaborators)]

    script_dir = os.path.dirname(os.path.realpath(__file__))

    connection = ZMQServer('{} connection'.format(agg_id), server_addr='*', server_port=server_port)

    with open(os.path.join(script_dir, '..', 'initial_models', "{}.pbuf".format(initial_model)), "rb") as f:
        loaded = f.read()
    model = ModelProto().FromString(loaded)

    agg = Aggregator(agg_id, fed_id, col_ids, connection, model)

    agg.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_collaborators', '-n', type=int, default=4)
    parser.add_argument('--initial_model', '-i', type=str, default='PyTorchMNISTCNN')
    parser.add_argument('--server_port', '-sp', type=int, default=5555)
    args = parser.parse_args()
    main(**vars(args))
