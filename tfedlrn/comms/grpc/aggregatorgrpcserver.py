# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import grpc
import signal
from concurrent import futures
import multiprocessing
import os
import logging
import time

from ...proto import datastream_to_proto, proto_to_datastream
from ...proto.lowlevelstrawman_pb2_grpc import AggregatorServicer, add_AggregatorServicer_to_server

class AggregatorGRPCServer(AggregatorServicer):
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def validate_collaborator(self, request, context):
        if not self.disable_tls:
            common_name = context.auth_context()['x509_common_name'][0].decode("utf-8")
            collaborator_common_name = request.header.sender
            if not self.aggregator.valid_collaborator_CN_and_id(common_name, collaborator_common_name):
                raise ValueError("Invalid collaborator. CN: |{}| collaborator_common_name: |{}|".format(common_name, collaborator_common_name))

    #def RequestJob(self, request, context):
    #    self.validate_collaborator(request, context)
    #    return self.aggregator.RequestJob(request)

    def GetTasks(self, request, context):
        self.validate_collaborator(request, context)
        return self.aggregator.GetTasks(request)

    def GetAggregatedTensor(self, request, context):
        self.validate_collaborator(request, context)
        return self.aggregator.GetAggregatedTensor(request)

    def SendLocalTaskResults(self, request, context):
        proto = TasksResults()
        proto = datastream_to_proto(proto, request)
        self.validate_collaborator(proto, context)
        # turn data stream into local model update
        return self.aggregator.UploadLocalModelUpdate(proto)


    #def DownloadModel(self, request, context):
    #    self.validate_collaborator(request, context)
    #    # turn global model update into data stream
    #    proto = self.aggregator.DownloadModel(request) 
    #    return proto_to_datastream(proto, self.logger)

    #def UploadLocalModelUpdate(self, request, context):
    #    proto = LocalModelUpdate()
    #    proto = datastream_to_proto(proto, request)
    #    self.validate_collaborator(proto, context)
    #    # turn data stream into local model update
    #    return self.aggregator.UploadLocalModelUpdate(proto)

    #def UploadLocalMetricsUpdate(self, request, context):
    #    self.validate_collaborator(request, context)
    #    return self.aggregator.UploadLocalMetricsUpdate(request)

    def serve(self, 
              agg_port, 
              disable_tls=False, 
              disable_client_auth=False, 
              ca=None, 
              certificate=None, 
              private_key=None, 
              **kwargs):
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
        kwargs              : dict
            Not currently used
        """
        logger = logging.getLogger(__name__)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()), 
                             options=[('grpc.max_metadata_size', 32 * 1024 * 1024),
                                      ('grpc.max_send_message_length', 128 * 1024 * 1024),
                                      ('grpc.max_receive_message_length', 128 * 1024 * 1024)])
        add_AggregatorServicer_to_server(self, server)
        uri = "[::]:{port:d}".format(port=agg_port)
        self.disable_tls = disable_tls
        self.logger = logger

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
        
        logger.info('Starting aggregator.')
        server.start()
        try:
            while not self.aggregator.all_quit_jobs_sent():
                time.sleep(5)
        except KeyboardInterrupt:
            pass
        server.stop(0)
