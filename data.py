""" Survey Data to diagnose and predict the risk of diabetes mellitus """

# Authors: Juan C. Vidal <juan.vidal@usc.es>
#          Marcos Matabuena <mmatabuena@hsph.harvard.edu>
#
# License: GNU General Public License, version 3 (or shortly GPL-3 license)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight


class CustomDataset(Dataset):
    """
    CustomDataset is a subclass of the Dataset class from PyTorch's data utilities. It is designed to encapsulate and provide structured access to a dataset consisting of features (x), target labels (y), and weights (w). This class is particularly useful for loading and accessing data in a format compatible with PyTorch's machine learning models and training procedures.

    :param x (numpy array): 
        A numpy array representing the features of the dataset. Each row corresponds to an observation, and each column represents a feature.
    :param y (numpy array): 
        A numpy array representing the target labels or outputs corresponding to each observation in x.
    :param w (numpy array): 
        A numpy array representing weights associated with each observation in the dataset.
    """

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
    """
    The num_to_onehot function converts a given array of numerical values into a one-hot encoded format. This process involves creating a binary matrix where each column corresponds to a unique value in the input array. The function identifies these unique values, sorts them, and encodes each observation in the input as a one-hot vector.

    :param sample (numpy array): 
        A numpy array containing numerical values that need to be one-hot encoded. The array can be either one-dimensional or two-dimensional. If it is two-dimensional, it is squeezed into a one-dimensional array for processing.
    :returns:
        new_array (numpy array): A two-dimensional numpy array where each row corresponds to an observation in the input sample array, and each column corresponds to a unique value found in sample. For each row, exactly one entry is 1, indicating the presence of the corresponding category in the original sample, and all other entries are 0.
    """

    if len(sample.shape) > 1:
        sample = np.squeeze(sample)
    categories = np.sort(np.unique(sample))
    new_array = np.zeros((sample.shape[0], len(categories)))
    for s in range(len(categories)):
        index = (sample == categories[s])
        new_array[index, s] = 1
    return new_array


def normalize_column(column):
    """
    The normalize_column function normalizes a given column of numerical data to a range between 0 and 1. This normalization is achieved by subtracting the minimum value of the column from each element and then dividing by the range of the column (max - min), with a small epsilon added to prevent division by zero. This type of normalization is often used in data preprocessing to bring different features to a similar scale, enhancing the performance of many machine learning algorithms.

    :param column (pandas Series or numpy array): 
        A column of numerical data to be normalized. This can be a pandas Series as typically obtained from a pandas DataFrame column.
    :returns:
        feature_vector (numpy array): The resulting normalized column where each original value is scaled to a value between 0 and 1. The scaling is done based on the minimum and maximum values in the input column.
    """

    min_c = column.min()
    max_c = column.max()
    eps = np.finfo(float).eps
    feature_vector = (column.values - min_c) / (max_c - min_c + eps)
    return feature_vector


def load_data(file, variables, output_var):
    """
    This function loads and processes data from a specified CSV file. 
    It performs normalization and one-hot encoding on specified variables, 
    concatenates the processed features, and prepares various other elements 
    for further analysis. The function is specifically tailored for handling 
    datasets with a mix of integer and categorical variables.

    :param file (str): 
        Path to the CSV file containing the data.
    :param variables (list of str):
        List of variable names to be included in the analysis. These variables are expected to be a subset of predefined integer and categorical variables.
    :param output_var (str): 
        Name of the output variable in the dataset. This variable is used for creating the response vector y.
    :returns: data (dict): 
        A dictionary containing several key-value pairs:
            - seqn: Numpy array of sequence numbers extracted from the 'SEQN' column in the dataset.
            - x: Numpy array of processed feature vectors, where integer variables are normalized and categorical variables are one-hot encoded.
            - y: Numpy array of the output variable, reshaped for compatibility with neural network loss functions like nn.BCELoss.
            - w: Numpy array derived from the 'wtmec4yr_adj_norm' column of the dataset, reshaped similarly to y.
            - z: Numpy array of random values drawn from a standard normal distribution, matching the length of y.
            - sigma: A constant (float), set to 0.01 in this implementation.
            - colnames: List of column names after processing, which includes original names for integer variables and modified names for one-hot encoded categorical variables.
    """
    # Load csv
    df = pd.read_csv(file, index_col=0)

    L = []
    colnames = []
    int_variables = ['INQ020_7', 'INDFMMPI_7', 'WHD050_30', 'alcoholfrecuencia', 'RIDAGEYR.x', 'BMXHT', 'BMXWT',
                     'BMXBMI', 'BMXWAIST', 'LBXTR_64', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'BPXDI1', 'LBDSCHSI_43',
                     'LBXSTR_43', 'LBXSGL_43', 'LBXGH_39']

    cat_variables = ['BPQ020_40', 'HIQ011_1', 'HOQ065_13', 'KIQ026_19', 'MCQ160K_35', 'MCQ160N_35', 'MCQ220_35',
                     'MCQ365A_35', 'MCQ365D_35', 'MCQ365B_35', 'SLQ050_9', 'MCQ220', 'RIAGENDR', 'RIDRETH3']

    all_variables = int_variables + cat_variables

    for i, c in enumerate(variables):
        assert c in all_variables, "Variable {} not included".format(c)

        if c in int_variables:
            # Normalize to 0-1 range
            feature_vector = normalize_column(df[c])
            L.extend([np.expand_dims(feature_vector, 1)])
            colnames.extend([c])

        elif c in cat_variables:
            # One-hot encoding for categorical variables
            feature_vector = num_to_onehot(df[c].to_numpy())
            L.extend([feature_vector])
            for j in range(feature_vector.shape[1]):
                colnames.extend([c + "_" + str(j)])

    seqn = df['SEQN'].to_numpy()
    x = np.hstack(L).astype('float32')
    y = df[output_var].values.astype('float32').reshape(-1, 1)  # nn.BCELoss()
    w = 1/df['wtmec4yr_adj_norm'].values.astype('float32').reshape(-1, 1)

    # z = np.random.uniform(0, 1, len(y)).reshape(-1, 1)
    z = np.random.standard_normal(len(y)).reshape(-1, 1)
    sigma = 0.01

    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    data = {
        'seqn': seqn,
        'x': x,
        'y': y,
        'w': w,
        'z': z,
        'sigma': sigma,
        'colnames': colnames,
    }

    return data

