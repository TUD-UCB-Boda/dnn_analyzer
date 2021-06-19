def round_units(value: int, types: str, decimal: bool = False) -> str:
    """
    Rounds the passed values to the nearest unit and
    appends a unit description.

    :param decimal: true if conversion to decimal is desired
    :param value: value to be rounded
    :param types: type of unit to be appended
    :return: string of rounded value and appended unit
    """
    if decimal:
        basis = 1000
    else:
        basis = 1024

    if 0 < (value // basis ** 4):
        return str(round((value / basis ** 4), 2)) + " T" + types
    elif 0 < (value // basis ** 3):
        return str(round((value / basis ** 3), 2)) + " G" + types
    elif 0 < (value // basis ** 2):
        return str(round((value / basis ** 2), 2)) + " M" + types
    elif 0 < (value // basis):
        return str(round((value / basis), 2)) + " K" + types
    else:
        return str(round(value, 2)) + types


def round_mega(value: int, decimal: bool = False) -> str:
    """
    Used instead of round_units when conversion to mega unit
    is desired.
    :param value: value to be rounded
    :param decimal: true if conversion to decimal is desired
    :return: string of rounded value
    """
    if decimal:
        basis = 1000
    else:
        basis = 1024

    return str(round((value / basis ** 2), 3))
