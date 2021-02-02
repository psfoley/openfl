from openfl.federated.types import TypeHandler
import torch.nn as nn
from torch.optim import Optimizer
from openfl.federated.types import PyTorchModuleTypeHandler,PyTorchOptimizerTypeHandler, FloatTypeHandler

class TypeHandlerFactory:

    def __init__(self):
        pass

    def is_supported(self,attr):
        """Does the attribute have a type handler?"""
        if isinstance(attr,(nn.Module,Optimizer,float)):
            return True
        return False

    def get_type_handler(self,attr):
        if isinstance(attr,nn.Module):
            return PyTorchModuleTypeHandler()
        elif isinstance(attr,Optimizer):
            return PyTorchOptimizerTypeHandler()
        elif isinstance(attr,float):
            return FloatTypeHandler()
        else:
            raise ValueError(f'{type(attr)} does not have a supported TypeHandler')


