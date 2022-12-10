"""Utility methods."""


def rgb_to_hex(c):
    # c in the range [0..1]
    a = int(c * 255)
    return "#%02x%02x%02x" % (a, a, a)


def parse_float(v, default=1.):
    try:
        result = float(v)
        return result
    except:
        return default


def parse_int(v, default=1):
    try:
        result = int(v)
        return result
    except:
        return default
