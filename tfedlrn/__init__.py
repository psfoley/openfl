def check_type(obj, expected_type):
    if not isinstance(obj, expected_type):
        raise TypeError("Expected type {}, got type {}".format(type(obj), str(expected_type)))


def check_equal(x, y):
    if x != y:
        raise ValueError("{} != {}".format(x, y))
