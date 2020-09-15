# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import grpc
import logging

from fledge.protocols import datastream_to_proto, proto_to_datastream
from fledge.protocols import AggregatorStub

from logging import getLogger

logger = getLogger(__name__)

class CollaboratorGRPCClient():
    """Collaboration over gRPC-TLS."""

    def __init__(self,
                 agg_addr,
                 agg_port,
                 disable_tls,
                 disable_client_auth,
                 ca,
                 certificate,
                 private_key,
                 **kwargs):

        self.uri                 = f'{agg_addr}:{agg_port}'
        self.disable_tls         = disable_tls
        self.disable_client_auth = disable_client_auth
        self.ca                  = ca
        self.certificate         = certificate
        self.private_key         = private_key

        self.channel_options     = [('grpc.max_metadata_size',           32 * 1024 * 1024),
                                    ('grpc.max_send_message_length',    128 * 1024 * 1024),
                                    ('grpc.max_receive_message_length', 128 * 1024 * 1024)]

        if  self.disable_tls:
            self.channel = self.create_insecure_channel(self.uri)
        else:
            self.channel = self.create_tls_channel(self.uri, self.ca, self.disable_client_auth, self.certificate, self.private_key)

        logger.debug('Connecting to gRPC at {uri}')

        self.stub = AggregatorStub(self.channel)

    def create_insecure_channel(self, uri):
        """
        Sets an insecure gRPC channel (i.e. no TLS) if desired (warns user that this is not recommended)

        Args:
            uri: The uniform resource identifier fo the insecure channel

        Returns:
            An insecure gRPC channel object

        """
        logger.warn("gRPC is running on insecure channel with TLS disabled.")

        return grpc.insecure_channel(uri, options = self.channel_options)

    def create_tls_channel(self, uri, ca, disable_client_auth, certificate, private_key):
        """
        Sets an secure gRPC channel (i.e. TLS)

        Args:
            uri: The uniform resource identifier fo the insecure channel
            ca: The Certificate Authority filename
            disable_client_auth (boolean): True disabled client-side authentication (not recommended, throws warning to user)
            certificate: The client certficate filename from the collaborator (signed by the certificate authority)

        Returns:
            An insecure gRPC channel object
        """

        with open(ca, 'rb') as f:
            root_certificates = f.read()

        if  disable_client_auth:
            logger.warn('Client-side authentication is disabled.')
            private_key = None
            client_cert = None
        else:
            with open(private_key, 'rb') as f: private_key = f.read()
            with open(certificate, 'rb') as f: client_cert = f.read()

        credentials = grpc.ssl_channel_credentials(
           root_certificates = root_certificates,
           private_key       = private_key,
           certificate_chain = client_cert
        )

        return grpc.secure_channel(uri, credentials,options = self.channel_options)

    def GetTasks(self, message):
        return self.stub.GetTasks(message)

    def GetAggregatedTensor(self, message):
        return self.stub.GetAggregatedTensor(message)

    def SendLocalTaskResults(self, message):
      # convert (potentially) long list of tensors into stream
        stream  = []
        stream += proto_to_datastream(message, logger)

        return self.stub.SendLocalTaskResults(iter(stream))
