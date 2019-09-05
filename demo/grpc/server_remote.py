# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging
import signal

import grpc

import helloworld_pb2
import helloworld_pb2_grpc


class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        return helloworld_pb2.HelloReply(message='Hello, %s!' % request.name)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)

    # TLS stuff
    with open('out/10.24.14.200.key', 'rb') as f:
        private_key = f.read()
    with open('out/10.24.14.200.crt', 'rb') as f:
        certificate_chain = f.read()
    server_credentials = grpc.ssl_server_credentials( ( (private_key, certificate_chain), ) )
    server.add_secure_port('[::]:4433', server_credentials)

    # Start the server.
    server.start()
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass
    server.stop(0)

def serve_mutual_tls():
    """Mutual TLS connection. """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)

    # TLS stuff
    with open('out/private/spr-gpu02.key', 'rb') as f:
        private_key = f.read()
    with open('out/spr-gpu02.crt', 'rb') as f:
        certificate_chain = f.read()
    with open('out/localhost.crt', 'rb') as f:
        client_cert = f.read()
        # It looks like we can load multiple client certificates in this way.
        # Or we can trust the root CA.
        # client_cert += client_cert
    server_credentials = grpc.ssl_server_credentials( ( (private_key, certificate_chain), ), root_certificates=client_cert,
        require_client_auth=True )
    server.add_secure_port('[::]:4433', server_credentials)

    server.start()
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        pass
    server.stop(0)

if __name__ == '__main__':
    logging.basicConfig()
    # serve()
    serve_mutual_tls()
