from .data_handler import DataHandler
import numpy as np
import inspect
from pathlib import Path
from importlib import import_module

class DataHandlerFactory:

    def __init__(self):
        self.data_handlers = []
        self.load_available_data_handlers()

    def load_available_data_handlers(self):
        """ Search the current directory to determine which data types can be supported. 

        This is necessary because users may not have 
        tensorflow or pytorch installed in their environment
        """
        from sys import path
        import pkgutil

        current_file = Path(__file__).resolve()
        root = current_file.parent.resolve()  # types root, containing type handlers
        path.append(str(root))

        for module in root.glob('*.py'):  # load command modules

            package = module.parent
            module = module.name.split('.')[0]

            if module.count('__init__') or module.count('data_handler_factory') or module == 'data_handler':
                continue

            handler_module = import_module(module, package)
            # Find class name in module
            data_handler = None
            for name, handler in inspect.getmembers(handler_module):
                if inspect.isclass(handler) and name != 'DataHandler':
                    data_handler = handler
                    break

            print(f'module name = {module}')
            print(f'data_handler = {data_handler}')

            dependencies = data_handler.get_dependencies()
            dependencies_available = True
            if len(dependencies) > 0:
                for dependency in dependencies:
                    if not pkgutil.find_loader(dependency):
                        dependencies_available = False
                        break

            if not dependencies_available:
                continue

            self.data_handlers.append(data_handler)


    def is_supported(self,attr):
        """Does the attribute have a type handler?"""
        for data_handler in self.data_handlers:
            if isinstance(attr,data_handler.type()):
                return True
        return False


    def get_data_handler(self,attr):
        for data_handler in self.data_handlers:
            if isinstance(attr,data_handler.type()):
                return data_handler()
        
        raise ValueError(f'{type(attr)} does not have a supported DataHandler')


