# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed
# evaluation license agreement between Intel Corporation and you.

import grpc

from fledge.protocols import proto_to_datastream
from fledge.protocols import AggregatorStub

from logging import getLogger

# Interceptor related imports
import time
from typing import Optional, Tuple


class ConstantBackoff:
    def __init__(self, reconnect_interval, logger):
        self.reconnect_interval = reconnect_interval
        self.logger = logger

    def sleep(self):
        self.logger.info("Attempting to connect to aggregator...")
        time.sleep(self.reconnect_interval)


class RetryOnRpcErrorClientInterceptor(
    grpc.UnaryUnaryClientInterceptor, grpc.StreamUnaryClientInterceptor
):
    def __init__(
        self,
        sleeping_policy,
        status_for_retry: Optional[Tuple[grpc.StatusCode]] = None,
    ):
        self.sleeping_policy = sleeping_policy
        self.status_for_retry = status_for_retry

    def _intercept_call(self, continuation, client_call_details, request_or_iterator):

        while True:
            response = continuation(client_call_details, request_or_iterator)

            if isinstance(response, grpc.RpcError):

                # If status code is not in retryable status codes
                if (
                    self.status_for_retry
                    and response.code() not in self.status_for_retry
                ):
                    return response

                self.sleeping_policy.sleep()
            else:
                return response

    def intercept_unary_unary(self, continuation, client_call_details, request):
        return self._intercept_call(continuation, client_call_details, request)

    def intercept_stream_unary(
        self, continuation, client_call_details, request_iterator
    ):
        return self._intercept_call(continuation, client_call_details, request_iterator)


class CollaboratorGRPCClient:
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

        self.uri = f'{agg_addr}:{agg_port}'
        self.disable_tls = disable_tls
        self.disable_client_auth = disable_client_auth
        self.ca = ca
        self.certificate = certificate
        self.private_key = private_key

        self.channel_options = [
            ('grpc.max_metadata_size', 32 * 1024 * 1024),
            ('grpc.max_send_message_length', 128 * 1024 * 1024),
            ('grpc.max_receive_message_length', 128 * 1024 * 1024)
        ]

        self.logger = getLogger(__name__)

        if self.disable_tls:
            self.channel = self.create_insecure_channel(self.uri)
        else:
            self.channel = self.create_tls_channel(
                self.uri,
                self.ca,
                self.disable_client_auth,
                self.certificate,
                self.private_key
            )

        self.logger.debug('Connecting to gRPC at {uri}')

        # Adding an interceptor for RPC Errors
        self.interceptors = (
            RetryOnRpcErrorClientInterceptor(
                sleeping_policy=ConstantBackoff(
                    logger=self.logger,
                    reconnect_interval=int(kwargs.get('client_reconnect_interval', 1)),),
                status_for_retry=(grpc.StatusCode.UNAVAILABLE,),
            ),
        )
        self.stub = AggregatorStub(
            grpc.intercept_channel(self.channel, *self.interceptors)
        )

    def create_insecure_channel(self, uri):
        """
        Sets an insecure gRPC channel (i.e. no TLS) if desired (warns user
         that this is not recommended)

        Args:
            uri: The uniform resource identifier fo the insecure channel

        Returns:
            An insecure gRPC channel object

        """
        self.logger.warn(
            "gRPC is running on insecure channel with TLS disabled.")

        return grpc.insecure_channel(uri, options=self.channel_options)

    def create_tls_channel(self, uri, ca, disable_client_auth,
                           certificate, private_key):
        """
        Sets an secure gRPC channel (i.e. TLS)

        Args:
            uri: The uniform resource identifier fo the insecure channel
            ca: The Certificate Authority filename
            disable_client_auth (boolean): True disabled client-side
             authentication (not recommended, throws warning to user)
            certificate: The client certficate filename from the collaborator
             (signed by the certificate authority)

        Returns:
            An insecure gRPC channel object
        """

        with open(ca, 'rb') as f:
            root_certificates = f.read()

        if disable_client_auth:
            self.logger.warn('Client-side authentication is disabled.')
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

        return grpc.secure_channel(
            uri, credentials, options=self.channel_options)

    def disconnect(self):
        """
        Close the gRPC channel
        """
        self.logger.debug(f'Disconnecting from gRPC server at {self.uri}')
        self.channel.close()

    def reconnect(self):
        """
        Create a new channel with the gRPC server
        """

        # channel.close() is idempotent. Call again here in case it wasn't issued previously
        self.disconnect()

        if self.disable_tls:
            self.channel = self.create_insecure_channel(self.uri)
        else:
            self.channel = self.create_tls_channel(
                self.uri,
                self.ca,
                self.disable_client_auth,
                self.certificate,
                self.private_key
            )

        self.logger.debug(f'Connecting to gRPC at {self.uri}')

        self.stub = AggregatorStub(
            grpc.intercept_channel(self.channel, *self.interceptors)
        )

    def _atomic_connection(func):
        def wrapper(self, *args, **kwargs):
            self.reconnect()
            message = func(self, *args, **kwargs)
            self.disconnect()
            return message
        return wrapper

    @_atomic_connection
    def GetTasks(self, message):
        return self.stub.GetTasks(message)

    @_atomic_connection
    def GetAggregatedTensor(self, message):
        return self.stub.GetAggregatedTensor(message)

    @_atomic_connection
    def SendLocalTaskResults(self, message):
        # convert (potentially) long list of tensors into stream
        stream = []
        stream += proto_to_datastream(message, self.logger)

        return self.stub.SendLocalTaskResults(iter(stream))
