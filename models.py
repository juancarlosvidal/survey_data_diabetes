""" Survey Data to diagnose and predict the risk of diabetes mellitus """

# Authors: Juan C. Vidal <juan.vidal@usc.es>
#          Marcos Matabuena <mmatabuena@hsph.harvard.edu>
#
# License: GNU General Public License, version 3 (or shortly GPL-3 license)

import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Opt 1
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        # Opt 2
        # nn.init.orthogonal_(m.weight)
        # nn.init.constant_(m.bias, 0)
        # Opt 3
        # nn.init.uniform_(m.weight)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear.apply(init_weights)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class BinaryClassificacionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Multilayer Perceptron for classification"""
        super(BinaryClassificacionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.BatchNorm1d(8),
            nn.Dropout(p=0.50),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.layers.apply(init_weights)

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Multilayer Perceptron for conformal inference"""
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Dropout(p=0.50),
            nn.Linear(hidden_dim, output_dim)
        )
        self.layers.apply(init_weights)

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

