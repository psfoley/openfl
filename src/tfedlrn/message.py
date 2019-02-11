from enum import Enum, auto


class MessageType(Enum):
    VERSION = auto()
    LAYER = auto()
    ROLE = auto()
    REPLY_CODE = auto()
    VERSION_REQUEST = auto()
    ROLE_REQUEST = auto()
    LAYER_REQUEST = auto()
    LAYER_UPDATE = auto()
    VALIDATION_RESULTS = auto()


class Message(object):

    def __init__(self,
                 message_type,
                 work_id,
                 sender,
                 recipient,
                 payload,
                 reply_code=None):

        self.message_type = message_type
        self.work_id = work_id
        self.sender = sender
        self.recipient = recipient
        self.reply_code = reply_code
        self.payload = payload

    def __repr__(self):
        return f'{self.message_type} {self.sender}->{self.recipient} {self.work_id} {self.reply_code} {type(self.payload)}'

    def __str__(self):
        return self.__repr__()
