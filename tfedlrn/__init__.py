# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from .utils import load_yaml, get_object, split_tensor_dict_for_holdouts


def check_type(obj, expected_type, logger):
    if not isinstance(obj, expected_type):
        exception = TypeError("Expected type {}, got type {}".format(type(obj), str(expected_type)))
        logger.exception(repr(exception))
        raise exception


def check_equal(x, y, logger):
    if x != y:
        exception = ValueError("{} != {}".format(x, y))
        logger.exception(repr(exception))
        raise exception


def check_not_equal(x, y, logger, name='None provided'):
    if x == y:
        exception = ValueError("Name {}. Expected inequality, but {} == {}".format(name, x, y))
        logger.exception(repr(exception))
        raise exception

def check_is_in(element, _list, logger):
    if element not in _list:
        exception = ValueError("Expected sequence memebership, but {} is not in {}".format(element, _list))
        logger.exception(repr(exception))
        raise exception

def check_not_in(element, _list, logger):
    if element in _list:
        exception = ValueError("Expected not in sequence, but {} is in {}".format(element, _list))
        logger.exception(repr(exception))
        raise exception
