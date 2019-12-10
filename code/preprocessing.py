import pandas as pd
import numpy as np
import torch

from typing import List
from numpy.random import choice, shuffle
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from scipy.stats import zscore


CHANNELS = ['0', '1', '2']
TRAIN = "_TRAIN"
TEST = "_TEST"


def slice_timeseries(X: np.ndarray,
                    length: int,
                    overlap: int = 0,
                    count: int = None):
    """
    Split time-series to set of small stime-series and create choose
    'count' of them.

    Args:
        length: length of taget ts.
        overlap: TODO
        count: the size of taget dataset.

    Returns:
        np.ndarray: dataset of time-series with length 'length'
    """
    assert length > overlap
    starts = range(0, X.shape[0] - length + 1, length-overlap)
    timeseries = np.array([X[start : start-overlap + length] for start in starts])
    timeseries = timeseries if not count\
                      else timeseries[choice(len(timeseries), count, False)]

    return timeseries


def split_to_sequence(X, k, w, warn=True):
    """
    Split the time-series into sequence of small time-series to use
    encoder for the sequence.

    Args:
        X: input time-series
        k: 2*k + 1 is the length of each out time-series
        w: step size for window

    Returns:
        np.ndarray: Splitted time-series dataset.
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


def prepare_data(X, y, k, w, device, test_size=0.05, valid_size=0.75):
    """
    Create datasets from time-series and split into train, test, valid

    Args:
        X: input time-series
        k: 2*k + 1 is the length of each out time-series
        w: step size for window
    
    Returns:
        DataLoader: train dataset
        DataLoader: test dataset
        DataLoader: valid dataset
    """
    ds = SplittedDataset(X, y, k, w, device=device)
    train_ds, test_ds, valid_ds = train_test_valid_split(ds, test_size, valid_size)

    train_set = DataLoader(train_ds, batch_size=1024, shuffle=True)
    test_set = DataLoader(test_ds, batch_size=1024, shuffle=True)
    valid_set = DataLoader(valid_ds, batch_size=256, shuffle=True)
    
    return train_set, test_set, valid_set


def get_dataset(data_path, length=100):
    df_train = pd.read_csv(data_path + TRAIN + ".txt", header=None, delim_whitespace=True)
    df_test = pd.read_csv(data_path + TEST + ".txt", header=None, delim_whitespace=True)
        
    df = pd.concat([df_test, df_train], ignore_index=True, sort=False)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    if np.isnan(X).any():
        timeseries = []
        labels = []
        for label, x in zip(y, X):
            ts = slice_timeseries(x[~np.isnan(x)], length)
            timeseries.extend(ts)
            labels.extend(np.repeat(label, len(ts)))

        X = np.asarray(timeseries)
        y = np.asarray(labels)
    else:
        X = slice_timeseries(X.swapaxes(1, 0), min(length, X.shape[-1])).swapaxes(1, 2)
        X = np.vstack(X)
        y = np.repeat(y, X.shape[0] // y.shape[0])
    
    assert not np.isnan(X).any(), "Stand from under! NAN in data!"

    return zscore(X, 1), y


class SplittedDataset(Dataset):
    def __init__(self, timeseries, labels, window_size, shift_size, device):

        self.window_size = window_size
        self.shift_size = shift_size
        self.device = device
        self.X = np.array(timeseries)
        self.X_splitted = \
            [split_to_sequence(x, self.window_size, self.shift_size, False) for x in timeseries]
        self.X_splitted = torch.tensor(self.X_splitted, device=self.device,
                                       dtype=torch.float)
        self.y = torch.tensor(labels, device=self.device, dtype=torch.float)

        self.labels = labels
        self.device = device
        assert len(self.X) == len(self.labels)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X_splitted[idx], self.X[idx], self.y[idx]
