import pandas as pd
import numpy as np

from typing import List
from numpy.random import choice, shuffle

CHANNELS = ['0', '1', '2']


def get_class_timeseries(label: int,
                         data: pd.DataFrame,
                         start_shift: int,
                         length: int = np.iinfo(int).max,
                         channels: List[str] = CHANNELS):

    X = data.loc[data.labels == label, ]
    X = X.loc[start_shift : start_shift+length, :]
    
    return X.values

def slice_timeseries(X: np.ndarray,
                    length: int,
                    overlap: int = 0,
                    count: int = None):
    
    assert length > overlap
    starts = range(0, X.shape[0] - length, length-overlap)
    timeseries = np.array([X[start : start-overlap + length, :] for start in starts])

    return timeseries if not count\
                      else timeseries[choice(len(timeseries), count, False)]


def split_to_sequence(X, k, w):
    """
    Split the timeseries into sequence of small time-series to use
    encoder on each of them.

    X: input time-series
    k: 2*k + 1 is the length of each out time-series
    w: step size for window
    """

    starts = range(k, len(X)-k+1, w)
    if starts[-1] + k < len(X):
        print("Last {0} points are not considered.".format(len(X) - starts[-1] - k))

    return [X[start-k : start+k] for start in starts]
