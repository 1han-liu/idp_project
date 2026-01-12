import configparser
from datetime import timedelta
from decimal import Decimal


class CustomConfigParser(configparser.ConfigParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        out = {}
        for s in self:
            d = {}
            for o in self[s]:
                d[o] = self.get(s, o)
            out[s] = d
        return out

    # custom converter functions
    def gettimedelta(self, section, option, *, raw=False, vars=None,
                     fallback=configparser._UNSET, **kwargs):
        return self._get_conv(section, option, self._convert_to_timedelta,
                              raw=raw, vars=vars, fallback=fallback, **kwargs)

    def _convert_to_timedelta(self, s: str):
        duration_string = s.lower()
        total_micro_seconds = Decimal('0')
        prev_num = []
        is_valid = False
        for i, c in enumerate(duration_string):
            if c.isalpha():
                if prev_num:
                    num = Decimal(''.join(prev_num))
                    if c == 'd':
                        is_valid = True
                        total_micro_seconds += num * 1000000 * 60 * 60 * 24
                    elif c == 'h':
                        is_valid = True
                        total_micro_seconds += num * 60 * 60
                    elif c == 'm':
                        if i+1 < len(duration_string):
                            if duration_string[i+1] == 's':
                                is_valid = True
                                total_micro_seconds += num * 1000
                        else:
                            is_valid = True
                            total_micro_seconds += num * 1000000 * 60
                    elif c == 's':
                        is_valid = True
                        total_micro_seconds += num * 1000000
                    elif c == 'u':
                        if i+1 < len(duration_string):
                            if duration_string[i+1] == 's':
                                is_valid = True
                                total_micro_seconds += num
                    prev_num = []
            elif c.isnumeric() or c == '.':
                prev_num.append(c)
        if not (is_valid or s == '0'):
            raise ValueError('Cannot parse time delta: %s' % s)
        return timedelta(microseconds=float(total_micro_seconds))


