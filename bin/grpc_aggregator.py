import grpc
import argparse
import signal
from concurrent import futures
import time
import yaml
import os
import logging

from tfedlrn.aggregator.aggregator import Aggregator
from tfedlrn.proto import message_pb2, message_pb2_grpc

from tfedlrn.proto.message_pb2 import Job, JobRequest, JobReply
from tfedlrn.proto.message_pb2 import JOB_DOWNLOAD_MODEL, JOB_QUIT, JOB_TRAIN, JOB_VALIDATE, JOB_YIELD
from tfedlrn.proto.message_pb2 import ModelDownloadRequest, GlobalModelUpdate
from tfedlrn.proto.message_pb2 import LocalModelUpdate, LocalModelUpdateAck
from tfedlrn.proto.message_pb2 import LocalValidationResults, LocalValidationResultsAck

from setup_logging import setup_logging

class FLtask(message_pb2_grpc.AggregatorServicer, Aggregator):
    def __init__(self, plan_path):
        id, fed_id, col_ids, connection, init_model_fpath, latest_model_fpath = self.load_plan(plan_path)
        Aggregator.__init__(self, id, fed_id, col_ids, connection, init_model_fpath, latest_model_fpath)

    def load_plan(self, path):
        """Read key information from the plan to start an aggregator. """
        with open(path, 'r') as f:
            plan = yaml.safe_load(f.read())
        fed_id = plan['federation']
        aggregator = plan['aggregator']
        agg_id = aggregator['id']
        
        initial_weights_fpath = aggregator['initial_weights']
        latest_weights_fpath = aggregator['latest_weights']
        # tfedlrn_version = aggregator['tfedlrn_version']

        # Replaced ZMQ with gRPC
        connection = None
        self.addr = aggregator['addr']
        self.port = aggregator['port']

        num_collaborators = int(aggregator['collaborators'])
        col_ids = ["{}".format(i) for i in range(num_collaborators)]

        return agg_id, fed_id, col_ids, connection, initial_weights_fpath, latest_weights_fpath

    def Query(self, request, context):
        """Reimplement Aggregator.run(). 
        Alternative solution: wrap the two lines into as class methods that we can override with any communication library.
            message = self.connection.receive()
            self.connection.send(reply)
        """
        self.logger.debug("Start the federation [%s] with aggeregator [%s]." % (self.fed_id, self.id))

        # receive a message
        message = getattr(request, request.WhichOneof('payload'))
        t = time.time()

        # # FIXME: validate that the message is for me
        # if not(message.header.recipient == self.id):
        #     self.logger.exception("Assertion failed: message.header.recipient == self.id")

        # # validate that the message is for my federation
        # if not(message.header.federation_id == self.fed_id):
        #     self.logger.exception("Assertion failed: message.header.federation_id == self.fed_id")

        # # validate that the sender is one of my collaborators
        # if not(message.header.sender in self.col_ids):
        #     self.logger.exception("Assertion failed: message.header.sender in self.col_ids")

        if isinstance(message, LocalModelUpdate):
            reply = self.handle_local_model_update(message)
        elif isinstance(message, LocalValidationResults):
            reply = self.handle_local_validation_results(message)
        elif isinstance(message, JobRequest):
            reply = self.handle_job_request(message)
        elif isinstance(message, ModelDownloadRequest):
            reply = self.handle_model_download_request(message)

        # do end of round check
        self.end_of_round_check()

        if not isinstance(reply, JobReply) or reply.job is not JOB_YIELD:
            print('aggregator handled {} in time {}'.format(message.__class__.__name__, time.time() - t))

        return message_pb2.FLMessage(**{reply.__class__.__name__.lower():reply})


def serve(task, disable_tls=False, disable_client_auth=False, ca=None, certificate=None, private_key=None):
    """Start a gRPC service.

    Parameters
    ----------
    task                : FLtask
        The gRPC service task.
    disable_tls         : bool
        To disable the TLS. (Default: False)
    disable_client_auth : bool
        To disable the client side authentication. (Default: False)
    ca                  : str
        File path to the CA certificate.
    certificate         : str
        File path to the server certificate.
    private_key         : str
        File path to the private key.
    """
    logger = logging.getLogger(__name__)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    message_pb2_grpc.add_AggregatorServicer_to_server(task, server)
    uri = "{addr:s}:{port:d}".format(addr=task.addr, port=task.port)

    if disable_tls:
        logger.warn('gRPC is running on insecure channel with TLS disabled.')
        server.add_insecure_port(uri)
    else:
        with open(private_key, 'rb') as f:
            private_key = f.read()
        with open(certificate, 'rb') as f:
            certificate_chain = f.read()
        with open(ca, 'rb') as f:
            root_certificates = f.read()

        require_client_auth = not disable_client_auth
        if not require_client_auth:
            logger.warn('Client-side authentication is disabled.')

        server_credentials = grpc.ssl_server_credentials(
            ( (private_key, certificate_chain), ),
            root_certificates=root_certificates,
            require_client_auth=require_client_auth,
        )
        server.add_secure_port(uri, server_credentials)

    server.start()
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass
    server.stop(0)


if __name__ == '__main__':
    """
    Examples:
    python bin/grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --disable_tls
    python bin/grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --disable_client_auth --ca=files/grpc/localhost.crt --certificate=files/grpc/spr-gpu02.jf.intel.com.crt --private_key=files/grpc/private/spr-gpu02.jf.intel.com.key
    python bin/grpc_aggregator.py --plan_path federations/plans/mnist_a.yaml --ca=files/grpc/localhost.crt --certificate=files/grpc/spr-gpu02.jf.intel.com.crt --private_key=files/grpc/private/spr-gpu02.jf.intel.com.key
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan_path', '-pp', type=str)
    parser.add_argument('--disable_tls', action='store_true')
    parser.add_argument('--ca', type=str)
    parser.add_argument('--certificate', type=str)
    parser.add_argument('--private_key', type=str)
    parser.add_argument('--disable_client_auth', action='store_true')
    args = parser.parse_args()

    setup_logging()

    task = FLtask(args.plan_path)
    serve(task,
        disable_tls=args.disable_tls,
        disable_client_auth=args.disable_client_auth,
        ca=args.ca, certificate=args.certificate,
        private_key=args.private_key
    )