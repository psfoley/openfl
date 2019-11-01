import grpc
import signal
from concurrent import futures
import os
import logging

from ..proto.collaborator_aggregator_interface_pb2_grpc import AggregatorServicer, add_AggregatorServicer_to_server

class AggregatorGRPCServer(AggregatorServicer):
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def RequestJob(self, request, context):
        """Pass through to wrapped aggregator. Pulls nothing from context."""
        return self.aggregator.RequestJob(request)

    def DownloadModel(self, request, context):
        """Pass through to wrapped aggregator. Pulls nothing from context."""
        return self.aggregator.DownloadModel(request)

    def UploadLocalModelUpdate(self, request, context):
        """Pass through to wrapped aggregator. Pulls nothing from context."""
        return self.aggregator.UploadLocalModelUpdate(request)

    def UploadLocalMetricsUpdate(self, request, context):
        """Pass through to wrapped aggregator. Pulls nothing from context."""
        return self.aggregator.UploadLocalMetricsUpdate(request)

    def serve(self, addr, port, disable_tls=False, disable_client_auth=False, ca=None, certificate=None, private_key=None):
        """Start an aggregator gRPC service.

        Parameters
        ----------
        fltask              : FLtask
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
        add_AggregatorServicer_to_server(self, server)
        uri = "{addr:s}:{port:d}".format(addr=addr, port=port)

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