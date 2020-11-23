# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed
# evaluation license agreement between Intel Corporation and you.

from .assigner import Assigner
from .random_grouped_assigner import RandomGroupedAssigner
from .static_grouped_assigner import StaticGroupedAssigner


__all__ = [
    'Assigner',
    'RandomGroupedAssigner',
    'StaticGroupedAssigner',
]
