import sys

_cnt = 0


def print_status(msg):
    global _cnt
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    print(ERASE_LINE + CURSOR_UP_ONE)
    msg = "\r[{0:>2}] - {1}".format(_cnt, msg)
    sys.stdout.write(msg)
    sys.stdout.flush()
    _cnt += 1
