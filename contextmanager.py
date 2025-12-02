import sys
import io
import os
from contextlib import contextmanager
@contextmanager

def suppress_stdout(suppress=True):
    """
    Control the print output
    suppress=True: prohibit print
    suppress=False: allow print
    """

    if suppress:
        # save the standard output
        old_stdout = sys.stdout
        # prohibit print
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            # recover the standard output
            sys.stdout = old_stdout
    else:
        yield

@contextmanager

def no_print():

    with open(os.devnull, 'w') as devnull:

        old_stdout = sys.stdout

        sys.stdout = devnull

        try:

            yield

        finally:

            sys.stdout = old_stdout