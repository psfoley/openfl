# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import os
import logging
import logging.config
import coloredlogs
import yaml

def setup_logging(path="logging.yaml", default_level='info'):
    logging_level_dict = {
     'notset': logging.NOTSET,
     'debug': logging.DEBUG,
     'info': logging.INFO,
     'warning': logging.WARNING,
     'error': logging.ERROR,
     'critical': logging.CRITICAL
    }

    default_level = default_level.lower()
    if default_level not in logging_level_dict:
        raise Exception("Not supported logging level: %s", default_level)
    default_level = logging_level_dict[default_level]

    if os.path.isfile(path):
        with open(path, 'r') as f:
            try:
                config = yaml.safe_load(f.read())
                # create directories (if needed) to hold file handler output files
                for handler_name in config["handlers"]:
                    handler_config = config["handlers"][handler_name]
                    if "filename" in handler_config:
                        file_path = os.path.abspath(handler_config["filename"])
                        logdir_path = os.path.split(file_path)[0]
                        if not os.path.exists(logdir_path):
                            print("Creating log directory: ", logdir_path)
                            os.makedirs(logdir_path) 
                logging.config.dictConfig(config)
                coloredlogs.install()
                print("Loaded logging configuration: %s" % path)
            except Exception as e:
                print("Trouble loading logging configuration with file [%s]." % path)
                print(e)
                logging.basicConfig(level=default_level)
                coloredlogs.install(level=default_level)
    else:
        print("Logging configuration file [%s] not found." % path)
        logging.basicConfig(level=default_level)
        coloredlogs.install(level=default_level)