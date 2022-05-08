"""
Path-and-Address
----------------

Functions for command-line server tools used by humans.

:copyright: (c) 2012-2016 by Joe Esposito.
:license: MIT, see LICENSE for more details.
"""

__version__ = '2.0.1'


from .parsing import resolve, split_address
from .validation import valid_address, valid_hostname, valid_port


__all__ = [
    'resolve', 'split_address', 'valid_address', 'valid_hostname',
    'valid_port',
]
