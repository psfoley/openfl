from enum import Enum, auto

class ReplyCode(Enum):
    STALE = auto()
    ERROR = auto()
    OK = auto()
