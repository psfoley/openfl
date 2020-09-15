# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from .federation_pb2 import ModelProto, MetadataProto
from .federation_pb2 import NamedTensor, DataStream
from .federation_pb2 import TasksRequest, TasksResponse, TaskResults
from .federation_pb2 import MessageHeader, Acknowledgement
from .federation_pb2 import TensorRequest, TensorResponse

from .federation_pb2_grpc import AggregatorServicer, add_AggregatorServicer_to_server
from .federation_pb2_grpc import AggregatorStub

from .utils          import *