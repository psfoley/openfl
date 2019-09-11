import argparse
from concurrent import futures
import grpc
import os
import yaml
import importlib

from setup_logging import setup_logging

from tfedlrn.proto import message_pb2, message_pb2_grpc
from tfedlrn.collaborator.collaborator import Collaborator

def load_model(code_name):
    module = importlib.import_module(code_name)
    model = module.get_model()
    return model

def parse_plan(plan_path):
    """Read key information from the plan to start an collaborator. """
    with open(plan_path, 'r') as f:
        plan = yaml.safe_load(f.read())

    fed_id = plan['federation']
    aggregator = plan['aggregator']
    agg_id = aggregator['id']

    # Replace ZMQ with gRPC
    connection = None
    addr = aggregator['addr']
    port = aggregator['port']
    code_name = plan['model']['name']
    wrapped_model = load_model("models."+code_name)
    model_version = -1
    polling_interval = 4
    opt_treatment = plan['collaborator']['opt_vars_treatment']

    # id, agg_id, fed_id, wrapped_model, connection, model_version, polling_interval=4, opt_treatment="AGG"
    return addr, port, agg_id, fed_id, wrapped_model, connection, model_version, polling_interval, opt_treatment

class CollaboratorGRPC(Collaborator):
    """Collaboration over gRPC-TLS."""
    def __init__(self, channel, *args):
        super(CollaboratorGRPC, self).__init__(*args)
        self.gRPC_stub = message_pb2_grpc.AggregatorStub(channel)

    def send_and_receive(self, message):
        # FIXME: the Collaborator class should have this line of zmq communication as a method that we can override with gPRC.
        fl_message = message_pb2.FLMessage(**{message.__class__.__name__.lower():message})
        fl_reply = self.gRPC_stub.Query(fl_message)
        reply = getattr(fl_reply, fl_reply.WhichOneof('payload'))
        # validate the message pair

        # check message is from my agg to me
        if not (reply.header.sender == self.agg_id and reply.header.recipient == self.id):
            self.logger.exception("Assertion failed: reply.header.sender == self.agg_id and reply.header.recipient == self.id")

        # check that the federation id matches
        if not (reply.header.federation_id == self.fed_id):
            self.logger.exception("Assertion failed: reply.header.federation_id == self.fed_id")

        # check that the counters match
        if not(reply.header.counter == self.counter):
            self.logger.exception("Assertion failed: reply.header.counter == self.counter")

        # increment our counter
        self.counter += 1

        return reply


def run(plan_path, col_id, enable_tls, certificate_folder, require_client_auth):
    """
    Parse the FL plan, extract: aggregator addr:port, collaborator parameters
        ignore connection
    col_id from program argument.
    """

    if enable_tls:
        print("Enabled TLS.")
        with open(os.path.join(certificate_folder, 'localhost.crt'), 'rb') as f:
            root_certificates = f.read()

        if require_client_auth:
            print("Required client auth.")
            with open(os.path.join(certificate_folder, 'private/10.24.14.200.key'), 'rb') as f:
                private_key = f.read()
            with open(os.path.join(certificate_folder, '10.24.14.200.crt'), 'rb') as f:
                client_cert = f.read()
            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=client_cert
            )
        else:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
            )

    addr, port, agg_id, fed_id, wrapped_model, connection, model_version, polling_interval, opt_treatment = parse_plan(plan_path)
    uri = "{addr:s}:{port:d}".format(addr=addr, port=port)

    if enable_tls:
        with grpc.secure_channel(uri, credentials) as channel:
            col = CollaboratorGRPC(channel, col_id, agg_id, fed_id, wrapped_model, connection, model_version, polling_interval, opt_treatment)
            col.run()
    else:
        with grpc.insecure_channel(uri) as channel:
            col = CollaboratorGRPC(channel, col_id, agg_id, fed_id, wrapped_model, connection, model_version, polling_interval, opt_treatment)
            col.run()


if __name__ == '__main__':
    """
    python bin/grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id 0
    python bin/grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id 0 --enable_tls --certificate_folder files/grpc/
    python bin/grpc_collaborator.py --plan_path federations/plans/mnist_a.yaml --col_id 0 --enable_tls --certificate_folder files/grpc/ --require_client_auth
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan_path', '-pp', type=str)
    parser.add_argument('--col_id', type=str)
    parser.add_argument('--enable_tls', action='store_true')
    parser.add_argument('--certificate_folder', type=str)
    parser.add_argument('--require_client_auth', action='store_true')
    args = parser.parse_args()

    setup_logging()

    if os.environ.get('https_proxy'):
        del os.environ['https_proxy']
    if os.environ.get('http_proxy'):
        del os.environ['http_proxy']

    run(args.plan_path, args.col_id, args.enable_tls, args.certificate_folder, args.require_client_auth)