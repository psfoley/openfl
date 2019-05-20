import abc
import zmq


class ZMQConnection(metaclass=abc.ABCMeta):

    def __init__(self,
                 name,
                 server_addr='127.0.0.1',
                 server_port=5555,
                 compression_func=None,
                 decompression_func=None,
                 receive_timeout=-1):
        self.name = name
        self.server_addr = f'tcp://{server_addr}:{server_port}'
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

        print(f'{self} sending {message}')
        if self.compression_func is not None:
            message = self.compression_func(message)
        self.socket.send_pyobj(message)

    def receive(self):
        if self.socket is None:
            self._connect()

        message = self.socket.recv_pyobj()
        if self.decompression_func is not None:
            message = self.decompression_func(message)
        print(f'{self} received {message}')
        return message

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.name}'

    def __str__(self):
        return self.__repr__()


class ZMQServer(ZMQConnection):

    def _connect(self):
        print(f'{self} binding to socket: {self.server_addr}')
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(self.server_addr)


class ZMQClient(ZMQConnection):

    def _connect(self):
        print(f'{self} connecting to server: {self.server_addr}')
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(self.server_addr)
