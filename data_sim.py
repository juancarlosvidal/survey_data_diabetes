import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight

from data import normalize_column, num_to_onehot


def load_data(file, variables):
    # Load csv
    df = pd.read_csv(file, index_col=0)

    L = []
    colnames = []
    all_variables = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"]

    for i, c in enumerate(variables):
        assert c in all_variables, "Variable {} not included".format(c)
        L.extend([np.expand_dims(df[c], 1)])
        colnames.extend([c])

    # seqn = df['SEQN'].to_numpy()
    x = np.hstack(L).astype("float32")
    y = df["Y"].values.astype("float32").reshape(-1, 1)  # nn.BCELoss()
    w = 1 / df["Prob"].values.astype("float32").reshape(-1, 1)
    # w = w/np.sum(w)
    # w = np.ones(len(y)).reshape(-1, 1)
    t = df["p"].values.astype("float32").reshape(-1, 1)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(df["Y"]), y=df["Y"]
    )

    # z = np.random.uniform(0, 1, len(y)).reshape(-1, 1)
    z = np.random.standard_normal(len(y)).reshape(-1, 1)
    sigma = 0.01

    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    data = {
        # 'seqn': seqn,
        "x": x,
        "y": y,
        "w": w,
        "t": t,
        "z": z,
        "colnames": colnames,
        "sigma": sigma,
    }

    return data
