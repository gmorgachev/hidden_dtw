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