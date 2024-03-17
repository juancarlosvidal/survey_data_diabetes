import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight


class CustomDataset(Dataset):
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        xt = torch.from_numpy(self.x[idx, :])
        yt = torch.from_numpy(self.y[idx])
        wt = torch.from_numpy(self.w[idx])
        return xt, yt, wt


def num_to_onehot(sample):
    # Get unique values
    if len(sample.shape) > 1:
        sample = np.squeeze(sample)
    categories = np.sort(np.unique(sample))
    new_array = np.zeros((sample.shape[0], len(categories)))
    for s in range(len(categories)):
        index = sample == categories[s]
        new_array[index, s] = 1
    return new_array


def normalize_column(column):
    min_c = column.min()
    max_c = column.max()
    eps = np.finfo(float).eps
    feature_vector = (column.values - min_c) / (max_c - min_c + eps)
    return feature_vector
