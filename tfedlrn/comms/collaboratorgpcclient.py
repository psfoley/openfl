# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import grpc
import logging

from ..proto.collaborator_aggregator_interface_pb2_grpc import AggregatorStub

class CollaboratorGRPCClient():
    """Collaboration over gRPC-TLS."""
    def __init__(self, addr, port, disable_tls, ca, disable_client_auth, certificate, private_key):
        self.logger = logging.getLogger(__name__)
        uri = "{addr:s}:{port:d}".format(addr=addr, port=port)

        self.channel_options=[('grpc.max_metadata_size', 32 * 1024 * 1024),
                              ('grpc.max_send_message_length', 128 * 1024 * 1024),
                              ('grpc.max_receive_message_length', 128 * 1024 * 1024)]

        if disable_tls:
            self.channel = self.create_insecure_channel(uri)
        else:
            self.channel = self.create_tls_channel(uri, ca, disable_client_auth, certificate, private_key)

        self.logger.debug("Connecting to gRPC at %s" % uri)
        self.stub = AggregatorStub(self.channel)
        
    def create_insecure_channel(self, uri):
        self.logger.warn("gRPC is running on insecure channel with TLS disabled.")
        return grpc.insecure_channel(uri, options=self.channel_options)

    def create_tls_channel(self, uri, ca, disable_client_auth, certificate, private_key):
        with open(ca, 'rb') as f:
            root_certificates = f.read()

        if disable_client_auth:
            self.logger.warn("Client-side authentication is disabled.")
            private_key = None
            client_cert = None
        else:
            with open(private_key, 'rb') as f:
                private_key = f.read()
            with open(certificate, 'rb') as f:
                client_cert = f.read()

        credentials = grpc.ssl_channel_credentials(
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=client_cert
        )
        return grpc.secure_channel(uri, credentials, options=self.channel_options)

    def RequestJob(self, message):
        return self.stub.RequestJob(message)

    def DownloadModel(self, message):
        return self.stub.DownloadModel(message)

    def UploadLocalModelUpdate(self, message):
        return self.stub.UploadLocalModelUpdate(message)

    def UploadLocalMetricsUpdate(self, message):
        return self.stub.UploadLocalMetricsUpdate(message)
