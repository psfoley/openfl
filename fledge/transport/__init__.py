# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .grpc import AggregatorGRPCServer
from .grpc import CollaboratorGRPCClient

__all__ = [
    'AggregatorGRPCServer',
    'CollaboratorGRPCClient'
]
