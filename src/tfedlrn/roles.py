from enum import Enum, auto


class Role(Enum):
    TRAIN = auto()
    VALIDATE = auto()
    DONE = auto()
