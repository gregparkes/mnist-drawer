"""Utility methods."""


def rgb_to_hex(c):
    # c in the range [0..1]
    a = int(c * 255)
    return "#%02x%02x%02x" % (a, a, a)
