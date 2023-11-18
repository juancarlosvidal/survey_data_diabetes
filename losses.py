""" Survey Data to diagnose and predict the risk of diabetes mellitus """

# Authors: Juan C. Vidal <juan.vidal@usc.es>
#          Marcos Matabuena <mmatabuena@hsph.harvard.edu>
#
# License: GNU General Public License, version 3 (or shortly GPL-3 license)

import torch
from torch import Tensor
from torch.nn.modules import Module


class CustomBCELoss(Module):
    """
    CustomBCELoss is a custom implementation of the Binary Cross-Entropy (BCE) loss function, typically used in binary classification tasks. This class modifies the standard BCE loss by incorporating sample weights, allowing for weighted loss calculation where each sample's contribution to the loss is scaled by its corresponding weight. This is particularly useful in scenarios where certain samples in a dataset are more important than others or in handling class imbalances.
    """

    def __init__(self) -> None:
        super(CustomBCELoss, self).__init__()

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        """
        Computes the weighted BCE loss.

        :param output (Tensor): 
            The predictions from the model, expected to be probabilities (values between 0 and 1).
        :param target (Tensor): 
            The ground truth labels, with the same shape as output.
        :param weights (Tensor): 
            A tensor of weights, with each element corresponding to the weight of a sample in the batch.
        :returns: 
            A single tensor value representing the weighted BCE loss for the batch.
        """
        weights = weights / torch.sum(weights)
        yp = torch.clip(output, 1e-7, 1 - 1e-7)
        term_0 = (1 - target) * torch.log(1 - yp + 1e-7)
        term_1 = target * torch.log(yp + 1e-7)
        return -torch.sum(weights * (term_0 + term_1))


class CustomMSELoss(Module):
    """
    CustomMSELoss is a PyTorch-based implementation of the Mean Squared Error (MSE) loss function, modified to include sample weights. This custom loss function is typically used for regression tasks, where the objective is to minimize the squared differences between the predicted and actual values. The incorporation of sample weights allows for differential weighting of errors on different samples, which can be particularly useful in datasets with varying levels of importance for each sample.
    """
    def __init__(self) -> None:
        super(CustomMSELoss, self).__init__()

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        """
        Computes the weighted mean squared error.
        :param output (Tensor): 
            The predictions from the model.
        :param target (Tensor): 
            The ground truth values, with the same shape as output.
        :param weights (Tensor):    
            A tensor of weights, where each element corresponds to the weight of a sample in the batch.
        :returns: 
            A single tensor value representing the weighted mean squared error for the batch.

        """
        weights = weights / torch.sum(weights)
        return torch.sum(weights * torch.pow(output - target, 2), axis=0)


class PinballLoss(Module):
    """
    PinballLoss, also known as Quantile Loss, is a custom loss function used primarily for quantile regression tasks. This class implements the Pinball Loss function with an additional feature to incorporate sample weights, which can be particularly useful in cases where different samples have varying levels of importance.
    
    :param alpha (float): 
        A hyperparameter that determines the quantile to be estimated. The value of alpha is between 0 and 1, where alpha = 0.5 corresponds to median regression.
    """
    def __init__(self, alpha) -> None:
        super(PinballLoss, self).__init__()
        self.alpha = alpha

    def forward(self, yh: Tensor, y: Tensor, weights: Tensor) -> Tensor:
        """
        Computes the weighted BCE loss.

        :param yh (Tensor): 
            The predicted values from the model.
        :param y (Tensor): 
            The ground truth values, with the same shape as yh.
        :param weights (Tensor): 
            A tensor of weights, where each element corresponds to the weight of a sample in the batch.
        :returns: 
            A single tensor value representing the weighted Pinball Loss for the batch.
        """
        difference = y - yh
        weighted_difference = difference * weights
        loss_positive = self.alpha * weighted_difference[difference >= 0]
        loss_negative = (1 - self.alpha) * -weighted_difference[difference < 0]
        loss_sum = torch.sum(loss_positive) + torch.sum(loss_negative)
        return loss_sum


class TitledLoss(Module):
    """
    TitledLoss is a custom implementation of a quantile-based loss function, similar to the Pinball Loss. It is designed for quantile regression tasks, where the objective is to predict a specific quantile of the target variable distribution. This class extends the functionality by incorporating sample weights, allowing for a weighted loss calculation that can prioritize certain samples over others.
    
    :param quantile (float): 
        A hyperparameter specifying the target quantile for the regression. The value is between 0 and 1, where, for example, quantile = 0.5 would correspond to median regression.
    """
    def __init__(self, quantile) -> None:
        super(TitledLoss, self).__init__()
        self.quantile = quantile

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        """
        Computes the weighted quantile loss.

        :param output (Tensor): 
            The predicted values from the model.
        :param target (Tensor): 
            The ground truth values, with the same shape as output.
        :param weights (Tensor): 
            A tensor of weights, where each element corresponds to the weight of a sample in the batch.
        :returns: 
            A single tensor value representing the weighted quantile loss for the batch.
        """
        weights = weights / torch.sum(weights)
        e = (target - output)
        return torch.mean(weights * torch.maximum(self.quantile * e, (self.quantile - 1) * e))
