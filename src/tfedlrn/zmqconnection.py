import abc
import zmq

from tfedlrn.proto.message_pb2 import *


def serialize(message):
    flm = FLMessage(**{message.__class__.__name__.lower():message})
    return flm.SerializeToString()


def deserialize(s):
    flmessage = FLMessage.FromString(s)
    return getattr(flmessage, flmessage.WhichOneof('payload'))


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

        print('{} sending {}'.format(self, message.__class__.__name__))
        # if self.compression_func is not None:
        #     message = self.compression_func(message)
        message = serialize(message)
        self.socket.send(message)

    def receive(self):
        if self.socket is None:
            self._connect()

        message = self.socket.recv()
        # if self.decompression_func is not None:
        #     message = self.decompression_func(message)
        message = deserialize(message)
        print('{} received {}'.format(self, message.__class__.__name__))
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
