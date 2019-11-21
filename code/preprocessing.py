import pandas as pd
import numpy as np
import torch

from typing import List
from numpy.random import choice, shuffle
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

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

    timeseries =  timeseries if not count\
                      else timeseries[choice(len(timeseries), count, False)]

    return timeseries.tolist()


def split_to_sequence(X, k, w, warn=True):
    """
    Split the timeseries into sequence of small time-series to use
    encoder on each of them.

    X: input time-series
    k: 2*k + 1 is the length of each out time-series
    w: step size for window
    """

    starts = range(k, len(X)-k+1, w)
    if starts[-1] + k < len(X) and warn:
        print("Last {0} points are not considered.".format(len(X) - starts[-1] - k))

    return [X[start-k : start+k] for start in starts]


def train_test_valid_split(dataset: TensorDataset, test_size: float, valid_size: float):
    test_size = int(len(dataset) * test_size)
    valid_size = int(len(dataset) * valid_size)
    train_size = len(dataset) - test_size - valid_size
    assert train_size > 0, "Invalid size of train set"
    assert len(dataset) == test_size + valid_size + train_size, "Invalid sum of sizes"
    
    return random_split(dataset, [train_size, test_size, valid_size])


class SplittedDataset(Dataset):

    def __init__(self, timeseries, labels, window_size, shift_size, device):

        self.window_size = window_size
        self.shift_size = shift_size
        self.device = device
        self.X = np.array(timeseries)
        self.X_splitted = \
            [split_to_sequence(x, self.window_size, self.shift_size, False) for x in timeseries]
        self.X_splitted = torch.tensor(self.X_splitted, device=self.device, dtype=torch.float)
        self.y = torch.tensor(labels, device=self.device, dtype=torch.float)

        self.labels = labels
        self.device = device
        assert len(self.X) == len(self.labels)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X_splitted[idx], self.X[idx], self.y[idx]
