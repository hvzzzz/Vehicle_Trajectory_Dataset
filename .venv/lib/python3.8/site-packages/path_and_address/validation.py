import re


_hostname_re = re.compile('(?!-)[A-Z\d-]{1,63}(?<!-)$', re.IGNORECASE)


def valid_address(address):
    """
    Determines whether the specified address string is valid.
    """
    if not address:
        return False

    components = str(address).split(':')
    if len(components) > 2 or not valid_hostname(components[0]):
        return False

    if len(components) == 2 and not valid_port(components[1]):
        return False

    return True


def valid_hostname(host):
    """
    Returns whether the specified string is a valid hostname.
    """
    if len(host) > 255:
        return False

    if host[-1:] == '.':
        host = host[:-1]

    return all(_hostname_re.match(c) for c in host.split('.'))


def valid_port(port):
    """
    Returns whether the specified string is a valid port,
    including port 0 (random port).
    """
    try:
        return 0 <= int(port) <= 65535
    except:
        return False
