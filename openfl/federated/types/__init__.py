# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Types package."""

from warnings import catch_warnings, simplefilter
import pkgutil

from .type_handler import TypeHandler  # NOQA
if pkgutil.find_loader('torch'):
    from .pytorch_module_type_handler import PyTorchModuleTypeHandler  # NOQA
    from .pytorch_optimizer_type_handler import PyTorchOptimizerTypeHandler  # NOQA
if pkgutil.find_loader('tensorflow'):
    from .keras_model_type_handler import KerasModelTypeHandler  #NOQA
from .float_type_handler import FloatTypeHandler  #NOQA
from .type_handler_factory import TypeHandlerFactory  # NOQA

