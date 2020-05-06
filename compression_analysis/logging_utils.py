import logging
from colorlog import ColoredFormatter
import os

logger = logging.getLogger('AVH')
logger.setLevel(logging.DEBUG)
logger.handlers = []
logger.propagate = False

if __name__=="__main__":
    print('logging ......')
    logger.warning('This is warning message')
    logger.debug('This is a debug message')
    logger.error('This is a error message')
    logging.error('This is a error message')
    logger.critical('This is a critical message')
    logger.info('This is a info message')
