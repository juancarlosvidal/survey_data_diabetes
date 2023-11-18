""" Survey Data to diagnose and predict the risk of diabetes mellitus """

# Authors: Juan C. Vidal <juan.vidal@usc.es>
#          Marcos Matabuena <mmatabuena@hsph.harvard.edu>
#
# License: GNU General Public License, version 3 (or shortly GPL-3 license)

import torch
import torch.nn as nn


def init_weights(m):
    """
    The init_weights function is designed to initialize the weights and biases of linear layers in a neural network. This function is typically used as an argument to the apply method of a PyTorch nn.Module, allowing custom weight initialization across all linear layers in the module. It includes several options for weight initialization, each of which is a common strategy in deep learning to optimize network performance and convergence.

    :param m (nn.Module): 
        A module from a PyTorch neural network. The function checks if this module is an instance of nn.Linear (a linear layer), and if so, applies weight initialization to it.
    """
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
    """
    LogisticRegression is a PyTorch implementation of the logistic regression model, a fundamental algorithm for binary classification tasks. This class constructs a simple neural network with a single linear layer followed by a sigmoid activation function, which maps the linear combination of inputs to a probability value between 0 and 1.
    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear.apply(init_weights)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        :param x (Tensor): 
            The input tensor containing features.
        :returns: 
            The probability predictions as a result of the logistic regression model.
        """
        outputs = torch.sigmoid(self.linear(x))
        return outputs


class BinaryClassificacionModel(nn.Module):
    """
    BinaryClassificacionModel is a custom PyTorch neural network model designed for binary classification tasks. It is a multilayer perceptron (MLP) with a structure that includes linear layers, activation functions, dropout for regularization, and a final sigmoid layer for output. This architecture is suitable for tasks where the goal is to predict binary outcomes (e.g., 0 or 1).
    The constructor initializes a sequential model (self.layers) comprising a linear layer, a ReLU activation function, a dropout layer, another linear layer, and finally a sigmoid activation function. It also applies the init_weights function to initialize the weights of the layers.
    
    :param input_dim (int): 
        The dimensionality of the input features.
    :param hidden_dim (int): 
        The number of neurons in the hidden layer.
    :param output_dim (int): 
        The dimensionality of the output, which is typically 1 for binary classification.
    """
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
        """
        Defines the forward pass of the model.

        :param x (Tensor): 
            The input tensor containing features.
        :returns: 
            The output tensor after passing through the layers of the model, representing the probability of the positive class in the binary classification task.
        """
        return self.layers(x)


class RegressionModel(nn.Module):
    """
    RegressionModel is a PyTorch-based neural network model designed for regression tasks. It is structured as a multilayer perceptron (MLP) and is tailored for conformal inference applications. The model architecture includes linear layers, a ReLU activation function, a dropout layer for regularization, but notably does not include a final activation function, allowing it to output continuous values suitable for regression.
    Initializes a sequential model (self.layers) comprising a linear layer, a ReLU activation function, a dropout layer for regularization, and a final linear layer. The model's weights are initialized using the init_weights function.

    :param input_dim (int): 
        The dimensionality of the input features.
    :param hidden_dim (int): 
        The number of neurons in the hidden layer.
    :param output_dim (int): 
        The dimensionality of the output, which corresponds to the number of target variables in the regression task.
    """
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
        """
        Defines the forward pass of the model.

        :param x (Tensor): 
            The input tensor containing features.
        :returns: 
            The output tensor after passing through the layers of the model, representing the continuous predictions for the regression task.        
        """
        return self.layers(x)

