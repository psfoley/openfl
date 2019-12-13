#!/usr/bin/env python3
import zmq
import pickle
import argparse
import numpy as np
import abc


class ZMQConnection(metaclass=abc.ABCMeta):

    def __init__(self,
                 name,
                 server_addr='127.0.0.1',
                 server_port=5555,
                 compression_func=None,
                 decompression_func=None,
                 receive_timeout=-1):
        self.name = name
        self.server_addr = 'tcp://{}:{}'.format(server_addr, server_port)
        self.context = zmq.Context()
        self.context.setsockopt(zmq.RCVTIMEO, receive_timeout)
        self.socket = None
        self.compression_func = compression_func
        self.decompression_func = decompression_func

    @abc.abstractmethod
    def _connect(self):
        pass

    def send(self, message):
        if self.socket is None:
            self._connect()

        if self.compression_func is not None:
            message = self.compression_func(message)
        self.socket.send_pyobj(message)

    def receive(self):
        if self.socket is None:
            self._connect()

        message = self.socket.recv_pyobj()
        if self.decompression_func is not None:
            message = self.decompression_func(message)
        return message

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.__repr__()


class ZMQServer(ZMQConnection):

    def _connect(self):
        print('{} binding to socket: {}'.format(self, self.server_addr))
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(self.server_addr)


class ZMQClient(ZMQConnection):

    def _connect(self):
        print('{} connecting to server: {}'.format(self, self.server_addr))
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.server_addr)


def server(n=3, port=5555, **kwargs):
    zmq_server = ZMQServer(__file__, server_port=port)
    for i in range(n):
        request = zmq_server.receive()
        print('Server received message number {} of size {}'.format(request['message_number'], 4 * request['array'].shape[0]))
        reply = op(request['array'])
        print('Server replying {}'.format(reply))
        zmq_server.send(reply)


def op(array):
    return np.mean(array)


def client(n=3, port=5555, message_size_mb=1, **kwargs):
    zmq_client = ZMQClient(__file__, server_port=port)
    for i in range(1, n + 1):
        array = np.arange(message_size_mb * 1024 * 1024 // 4).astype(np.float32) * i
        request = {
            'message_number': i,
            'array': array
            }

        zmq_client.send(request)
        print('Client {} sent message number {} of size {}'.format(zmq_client, request['message_number'], 4 * request['array'].shape[0]))
        reply = zmq_client.receive()
        print('Client {} received {}'.format(zmq_client, reply))
        print('Verified correct: {}'.format(op(array) == reply))


def main(mode='server', **kwargs):
    if mode == 'server':
        server(**kwargs)
    elif mode == 'client':
        client(**kwargs)
    else:
        raise ValueError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='server', choices=['server', 'client'])
    parser.add_argument('--n', '-n', type=int, default=3)
    parser.add_argument('--port', '-p', type=int, default=5555)
    parser.add_argument('--message_size_mb', '-s', type=int, default=1)
    main(**vars(parser.parse_args()))
