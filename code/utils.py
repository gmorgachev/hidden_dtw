import numpy as np
import matplotlib.pyplot as plt
import torch

from time import time
from sklearn.neighbors import KNeighborsClassifier
from dtaidistance import dtw, dtw_c, dtw_ndim
from IPython.display import clear_output
from sklearn.model_selection import train_test_split


def classify(timeseries, hiddens, labels):
    idxs = np.arange(len(hiddens)).reshape(-1, 1)
    
    scores_hidden = []
    scores_ts = []
    
    t = time()
    matrix_hidden = dtw_ndim.distance_matrix(hiddens)
    print("hidden_ts: {:.3f}".format(time() - t))

    t = time()
    matrix_ts = dtw.distance_matrix(timeseries, use_c=True)
    print("raw_ts: {:.3f}".format(time() - t))

    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(idxs, labels.cpu().numpy(), test_size=0.7)
        clf = KNeighborsClassifier(metric=get_metric(matrix_ts), algorithm="brute", n_neighbors=3)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores_ts.append(score)
        clf = KNeighborsClassifier(metric=get_metric(matrix_hidden), algorithm="brute", n_neighbors=3)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores_hidden.append(score)

    print("Raw ts score: {:.3f} +- {:.3f}".format(np.mean(scores_ts), np.std(scores_ts)))    
    print("Hidden ts score: {:.3f} +- {:.3f}".format(np.mean(scores_hidden), np.std(scores_hidden)))
    
    return scores_ts, scores_hidden
   

def plot(ts, ts_splitted, hiddens, out, id0):
    plt.figure(figsize=[16, 10])
    plt.subplot(2, 2, 1)
    plt.title("Raw")
    plt.plot(ts[id0])
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.title("Raw splitted")
    plt.plot(ts_splitted[id0])
    plt.grid()
    
    plt.subplot(2, 2, 3)
    plt.title("Hiddens")
    plt.plot(hiddens[id0])
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.title("Out")
    plt.plot(out[id0])
    plt.grid()


def _metric(id1, id2, matrix):
    """
    Return distance between elements from distance matrix
    """
    x, y = int(min(id1[0], id2[0])), int(max(id1[0], id2[0]))
    return matrix[x, y]


def get_metric(matrix):
    """
    Return distance function from distance matrix
    """
    return lambda id1, id2: _metric(id1, id2, matrix)
    

def valid(model, valid_ds, loss_fn):
    model.eval()
    with torch.no_grad():
        loss = 0.
        it = iter(valid_ds)
        for batch, _, _ in it:
            batch = batch.permute(1, 0, 2)
            out = model(batch)
            loss += loss_fn(batch, out)
    
    return loss.cpu().detach().numpy()


def train(model, train_ds, optim, loss_fn, valid_ds, n_step, info=None):
    model.train()
    history_train = []
    history_valid = []
    for step in range(n_step):
        it = iter(train_ds)
        train_loss = 0.
        for batch, _, _ in it:
            batch = batch.permute(1, 0, 2)
            out = model(batch)
            loss = loss_fn(batch, out)
            
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss += loss.cpu().detach().numpy()
        
        history_train.append(train_loss)
        if (step+1) % 100 == 0:
            clear_output(True)
            fig = plt.figure(figsize=[16, 5])
            fig.suptitle("Train {}".format(info), fontsize=16)
            plt.subplot(1, 2, 1)
            plt.title("Train loss")
            plt.plot(history_train)
            plt.grid()

            plt.subplot(1, 2, 2)
            plt.title("Valid loss")
            plt.plot(history_valid)
            plt.grid()
            
            plt.show()
        if step % 50 == 0:
            valid_loss = valid(model, valid_ds, loss_fn)
            history_valid.append(valid_loss)
            model.train()