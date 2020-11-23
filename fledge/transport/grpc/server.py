# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed
# evaluation license agreement between Intel Corporation and you.

from grpc import server, ssl_server_credentials
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from logging import getLogger
from time import sleep

from fledge.protocols import datastream_to_proto
from fledge.protocols import TaskResults
from fledge.protocols import AggregatorServicer, add_AggregatorServicer_to_server


class AggregatorGRPCServer(AggregatorServicer):
    """
    gRPC server class for the Aggregator
    """

    def __init__(self,
                 aggregator,
                 agg_port,
                 disable_tls=False,
                 disable_client_auth=False,
                 ca=None,
                 certificate=None,
                 private_key=None,
                 **kwargs):
        """
        Class initializer
        Args:
            aggregator: The aggregator
        Args:
            fltask (FLtask): The gRPC service task.
            disable_tls (bool): To disable the TLS. (Default: False)
            disable_client_auth (bool): To disable the client side authentication. (Default: False)
            ca (str): File path to the CA certificate.
            certificate (str): File path to the server certificate.
            private_key (str): File path to the private key.
            kwargs (dict): Additional arguments to pass into function
        """

        self.aggregator = aggregator
        self.uri = f'[::]:{agg_port}'
        self.disable_tls = disable_tls
        self.disable_client_auth = disable_client_auth
        self.ca = ca
        self.certificate = certificate
        self.private_key = private_key
        self.channel_options = [('grpc.max_metadata_size', 32 * 1024 * 1024),
                                ('grpc.max_send_message_length', 128 * 1024 * 1024),
                                ('grpc.max_receive_message_length', 128 * 1024 * 1024)]

        self.logger = getLogger(__name__)

    def validate_collaborator(self, request, context):
        """
        Validate the collaborator

        Args:
            request: The gRPC message request
            context: The gRPC context

        Raises:
            ValueError: If the collaborator or collaborator certificate is not valid then raises error.

        """
        if not self.disable_tls:
            common_name = context.auth_context()['x509_common_name'][0].decode('utf-8')
            collaborator_common_name = request.header.sender
            if not self.aggregator.valid_collaborator_CN_and_id(common_name, collaborator_common_name):
                raise ValueError(
                    f'Invalid collaborator. CN: |{common_name}| '
                    f'collaborator_common_name: |{collaborator_common_name}|')

    def GetTasks(self, request, context):
        """
        gRPC request for a job from aggregator

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        self.validate_collaborator(request, context)
        return self.aggregator.GetTasks(request)

    def GetAggregatedTensor(self, request, context):
        """
        gRPC request for a job from aggregator

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        self.validate_collaborator(request, context)
        return self.aggregator.GetAggregatedTensor(request)

    def SendLocalTaskResults(self, request, context):
        """
        gRPC request for a model download from aggregator

        Args:
            request: The gRPC message request
            context: The gRPC context

        """
        proto = TaskResults()
        proto = datastream_to_proto(proto, request)

        self.validate_collaborator(proto, context)

        # turn data stream into local model update
        return self.aggregator.SendLocalTaskResults(proto)

    def serve(self):
        """
        Start an aggregator gRPC service.
        """

        self.server = server(ThreadPoolExecutor(max_workers=cpu_count()),
                             options=self.channel_options)

        add_AggregatorServicer_to_server(self, self.server)

        if self.disable_tls:

            self.logger.warn('gRPC is running on insecure channel with TLS disabled.')

            self.server.add_insecure_port(self.uri)

        else:

            with open(self.private_key, 'rb') as f:
                private_key = f.read()
            with open(self.certificate, 'rb') as f:
                certificate_chain = f.read()
            with open(self.ca, 'rb') as f:
                root_certificates = f.read()

            if self.disable_client_auth:
                self.logger.warn('Client-side authentication is disabled.')

            self.server_credentials = ssl_server_credentials(((private_key, certificate_chain),),
                                                             root_certificates=root_certificates,
                                                             require_client_auth=not self.disable_client_auth)

            self.server.add_secure_port(self.uri, self.server_credentials)

        self.logger.info('Starting Aggregator gRPC Server')

        self.server.start()

        try:
            while not self.aggregator.all_quit_jobs_sent():
                sleep(5)
        except KeyboardInterrupt:
            pass

        self.server.stop(0)
