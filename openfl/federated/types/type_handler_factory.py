from openfl.federated.types import TypeHandler
from pathlib import Path
from importlib import import_module
import inspect

class TypeHandlerFactory:

    def __init__(self):
        self.type_handlers = []
        self.load_available_type_handlers()

    def load_available_type_handlers(self):
        """ Search the current directory to determine which types can be supported. 

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

            if module.count('__init__') or module.count('type_handler_factory') or module == 'type_handler':
                continue

            print(f'module = {module}, package = {package}')
            handler_module = import_module(module, package)
            # Find class name in module
            type_handler = None
            for name, handler in inspect.getmembers(handler_module):
                if inspect.isclass(handler):
                    type_handler = handler
                    break

            dependencies = type_handler.get_dependencies()
            dependencies_available = True
            if len(dependencies) > 0:
                for dependency in dependencies:
                    if not pkgutil.find_loader(dependency):
                        dependencies_available = False
                        break

            if not dependencies_available:
                continue

            self.type_handlers.append(type_handler)


    def is_supported(self,attr):
        """Does the attribute have a type handler?"""
        for type_handler in self.type_handlers:
            if isinstance(attr,type_handler.type()):
                return True
        return False

    def get_type_handler(self,attr):
        """Return the correct type handler for the attribute."""
        for type_handler in self.type_handlers:
            if isinstance(attr,type_handler.type()):
                return type_handler()

        raise ValueError(f'{type(attr)} does not have a supported TypeHandler')

