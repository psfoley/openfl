from enum import Enum, auto


class CollaboratorJob(Enum):
    TRAIN = auto()
    VALIDATE = auto()
    YIELD = auto()
    QUIT = auto()
    DOWNLOAD_MODEL = auto()


class MessageType(Enum):
    JOB_REQUEST = auto()
    JOB_REPLY = auto()
    TRAIN_UPDATE = auto()
    TRAIN_ACK = auto()
    VALIDATE_UPDATE = auto()
    VALIDATE_ACK = auto()
    MODEL_DOWNLOAD_REQUEST = auto()
    MODEL_DOWNLOAD_REPLY = auto()


# FIXME: payloads shouldn't just be magically known
class Message(object):

    def __init__(self,
                 message_type,
                 sender,
                 recipient,
                 payload):

        self.message_type = message_type
        self.sender = sender
        self.recipient = recipient
        self.payload = payload

    def __repr__(self):
        return '{} {}->{} {}'.format(self.message_type, self.sender, self.recipient, type(self.payload))

    def __str__(self):
        return self.__repr__()
