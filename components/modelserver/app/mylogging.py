import logging
import sys

def warning_filter(record: logging.LogRecord):
    if record.levelno >= 30:
        return False
    return True


debug_formatter = logging.Formatter(
    '%(asctime)s %(levelname)-7s %(module)-20s %(message)s')

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setFormatter(debug_formatter)
stderr_handler.setLevel(logging.WARNING)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(debug_formatter)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.addFilter(warning_filter)

logging.basicConfig(
    handlers=[stderr_handler, stdout_handler], level=logging.INFO)