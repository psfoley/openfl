#!/usr/bin/env python3
import argparse
import numpy as np

import tfedlrn
import tfedlrn.aggregator
from tfedlrn.aggregator.aggregator import Aggregator
from tfedlrn.zmqconnection import ZMQServer

from tfedlrn.proto.message_pb2 import *


def main(num_collaborators=4):
	agg_id = "notebook_agg"
	fed_id = "PT MNIST notebook fed"
	col_ids = ["notebook col {}".format(i) for i in range(num_collaborators)]

	connection = ZMQServer('notebook agg connection')

	with open("pytorch_mnist.pbuf", "rb") as f:
	    loaded = f.read()
	model = ModelProto().FromString(loaded)

	agg = Aggregator(agg_id, fed_id, col_ids, connection, model)

	agg.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_collaborators', '-n', type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))
