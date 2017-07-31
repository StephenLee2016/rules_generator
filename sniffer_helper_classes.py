from optparse import Option
from copy import copy

def _check_list(option, opt, value):
    try:
        return value.split(',')
    except ValueError:
        raise OptionValueError('option %s: invalid list value: %s' % (opt, value))

class ListOption(Option):
    TYPES = Option.TYPES + ('list',)
    TYPE_CHECKER = copy(Option.TYPE_CHECKER)
    TYPE_CHECKER['list'] = _check_list