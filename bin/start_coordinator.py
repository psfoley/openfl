#!/usr/bin/env python

"""Start a coordinator to run Federated Learning.

This program has two modes: the server mode and the client mode.

Example on MNIST with a single collaborator:
$ python bin/start_coordinator.py --mode=server --addr=* --port=6666 --plans=federations/plans/mnist_a.yaml
$ python bin/start_coordinator.py --mode=client --addr=127.0.0.1 --port=6666 --dataset=mnist_batch --software_version=1 --models_folder=/tmp/.flmodels/ --col_id=0

Example on MNIST with two collaborators:
$ python bin/start_coordinator.py --mode=server --addr=* --port=6666 --plans=federations/plans/mnist_b.yaml
$ python bin/start_coordinator.py --mode=client --addr=127.0.0.1 --port=6666 --dataset=mnist_batch --software_version=1 --models_folder=/tmp/.flmodels/ --splits 1 2 --split_index=0 --col_id=0
$ python bin/start_coordinator.py --mode=client --addr=127.0.0.1 --port=6666 --dataset=mnist_batch --software_version=1 --models_folder=/tmp/.flmodels/ --splits 1 2 --split_index=1 --col_id=1

Example on BraTS17 with two collaborators:
$ python bin/start_coordinator.py --mode=server --addr=* --port=6666 --plans=federations/plans/brats17_a.yaml
$ python bin/start_coordinator.py --mode=client --col_id=0 --addr=127.0.0.1 --port=6666 --dataset=BraTS17 --software_version=1 --models_folder=/tmp/.flmodels/
$ python bin/start_coordinator.py --mode=client --col_id=1 --addr=127.0.0.1 --port=6666 --dataset=BraTS17 --software_version=1 --models_folder=/tmp/.flmodels/
"""

import argparse
import os
import sys

from tfedlrn.coordinator.coordinator import Server, Client
from tfedlrn.zmqconnection import ZMQServer, ZMQClient

from setup_logging import setup_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str)
    parser.add_argument('--addr', '-a', type=str, default="*")
    parser.add_argument('--port', '-p', type=int, default=6666)

    parser.add_argument('--plans', type=str)

    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--software_version', '-v', type=str, default=os.environ['TFEDLRN_VER'])
    parser.add_argument('--models_folder', type=str, default="./models")
    parser.add_argument('--col_id', type=str)

    # Debug only.
    parser.add_argument('--splits', nargs='*')
    parser.add_argument('--split_index', type=int)
    args = parser.parse_args()

    setup_logging()

    if args.mode == "server":
        connection = ZMQServer('Server-Coordinator connection', server_addr=args.addr, server_port=args.port)
        coordinator = Server(connection, args.plans)
    elif args.mode == "client":
        if args.splits is not None:
            args.splits = [int(num) for num in args.splits]
        connection = ZMQClient('Client-Coordinator connection', server_addr=args.addr, server_port=args.port)
        coordinator = Client(connection, args.col_id, args.dataset, args.software_version, args.models_folder, splits=args.splits, split_idx=args.split_index)
    else:
        print("Please specify the mode: server or client.")
        sys.exit()
    coordinator.run()