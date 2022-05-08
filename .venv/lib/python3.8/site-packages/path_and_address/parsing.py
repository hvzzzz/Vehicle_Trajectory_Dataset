from .validation import valid_hostname, valid_port


def resolve(path_or_address=None, address=None, *ignored):
    """
    Returns (path, address) based on consecutive optional arguments,
    [path] [address].
    """
    if path_or_address is None or address is not None:
        return path_or_address, address

    path = None
    if split_address(path_or_address)[1] is not None:
        address = path_or_address
    else:
        path = path_or_address

    return path, address


def split_address(address):
    """
    Returns (host, port) with an integer port from the specified address
    string. (None, None) is returned if the address is invalid.
    """
    invalid = None, None

    if not address and address != 0:
        return invalid

    components = str(address).split(':')
    if len(components) > 2:
        return invalid

    if components[0] and not valid_hostname(components[0]):
        return invalid

    if len(components) == 2 and not valid_port(components[1]):
        return invalid

    if len(components) == 1:
        components.insert(0 if valid_port(components[0]) else 1, None)

    host, port = components
    port = int(port) if port else None

    return host, port
