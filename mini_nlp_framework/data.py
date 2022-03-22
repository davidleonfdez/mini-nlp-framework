from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Union


DEFAULT_BS = 64


class Vocab:
    def __init__(self, word_to_idx:Dict[str, int], idx_to_word:List[str]):
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word


def get_kfolds(X:np.array, y:np.array, seq_lengths:Union[List[int], np.array]=None, n:int=3):
    kf = KFold(n_splits=n, random_state=None, shuffle=False)
    splits = []
    if isinstance(seq_lengths, list): seq_lengths = np.array(seq_lengths)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sl_train, sl_test = ((seq_lengths[train_index], seq_lengths[test_index]) if seq_lengths is not None
                             else (None, None))
        splits.append((X_train, X_test, y_train, y_test, sl_train, sl_test))
    return splits


def get_dl_from_tensors(x, y, *extra_xs, bs=DEFAULT_BS):
    dataset = TensorDataset(x, y, *extra_xs)
    return DataLoader(dataset, batch_size=bs)


@dataclass
class DataLoaders:
    train:DataLoader=None
    valid:DataLoader=None
