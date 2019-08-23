import argparse
import os
import sys

from tfedlrn.coordinator.coordinator import Server, Client
from tfedlrn.zmqconnection import ZMQServer, ZMQClient

from setup_logging import setup_logging

"""
Examples:
python bin/start_coordinator.py --mode=server --addr=* --port=6666 --plans=federations/plans/mnist_a.yaml
python bin/start_coordinator.py --mode=client --addr=127.0.0.1 --port=6666 --dataset=mnist_batch --software_version=1 --models_folder=~/.flmodels/

python bin/start_coordinator.py --mode=server --addr=* --port=6666 --plans=federations/plans/
python bin/start_coordinator.py --mode=client --addr=127.0.0.1 --port=6666 --dataset=BraTS17 --software_version=1 --models_folder=~/.flmodels/

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str)
    parser.add_argument('--addr', '-a', type=str, default="*")
    parser.add_argument('--port', '-p', type=int, default=6666)

    parser.add_argument('--plans', type=str)

    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--software_version', '-v', type=str, default=os.environ['TFEDLRN_VER'])
    parser.add_argument('--models_folder', type=str, default="./models")
    args = parser.parse_args()

    setup_logging()

    if args.mode == "server":
        connection = ZMQServer('Server-Coordinator connection', server_addr=args.addr, server_port=args.port)
        coordinator = Server(connection, args.plans)
    elif args.mode == "client":
        connection = ZMQClient('Client-Coordinator connection', server_addr=args.addr, server_port=args.port)
        coordinator = Client(connection, args.dataset, args.software_version, args.models_folder)
    else:
        print("Please specify the mode: server or client.")
        sys.exit()
    coordinator.run()