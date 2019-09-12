import os
import logging
import logging.config
import coloredlogs
import yaml

def setup_logging(path="logging.yaml", default_level=logging.INFO):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            try:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
                coloredlogs.install()
                print("Loaded logging configuration: %s" % path)
            except Exception as e:
                print("Invalid logging configuration file [%s]." % path)
                print(e)
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        print("Logging configuration file [%s] not found." % path)
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)