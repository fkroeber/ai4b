import bz2
import numpy as np
import _pickle as cPickle
import sys, os
from contextlib import contextmanager


def compress_pickle(path, data):
    """Compress and save a pickled object to a file"""
    with bz2.BZ2File(path, "w") as f:
        cPickle.dump(data, f)


def decompress_pickle(file):
    """Load and decompress a pickled object from a file"""
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def normalise(x):
    return (x - x.min()) / (x.max() - x.min())


def standardise(x):
    return (x - x.mean()) / x.std()


def pad_edge(x):
    return np.pad(x[1:-1, 1:-1], pad_width=((1, 1), (1, 1)), constant_values=1)


@contextmanager
def suppress_stdout(verbose=False):
    """Suppress any output message/log."""
    if verbose:
        yield
    else:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
