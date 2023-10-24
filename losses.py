import torch
from torch import Tensor
from torch.nn.modules import Module


class CustomBCELoss(Module):
    def __init__(self) -> None:
        super(CustomBCELoss, self).__init__()

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        weights = weights / torch.sum(weights)
        yp = torch.clip(output, 1e-7, 1 - 1e-7)
        term_0 = (1 - target) * torch.log(1 - yp + 1e-7)
        term_1 = target * torch.log(yp + 1e-7)
        return -torch.sum(weights * (term_0 + term_1))


class CustomMSELoss(Module):
    def __init__(self) -> None:
        super(CustomMSELoss, self).__init__()

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        weights = weights / torch.sum(weights)
        return torch.sum(weights * torch.pow(output - target, 2), axis=0)


class PinballLoss(Module):
    def __init__(self, alpha) -> None:
        super(PinballLoss, self).__init__()
        self.alpha = alpha

    def forward(self, yh: Tensor, y: Tensor, weights: Tensor) -> Tensor:
        difference = y - yh
        weighted_difference = difference * weights
        loss_positive = self.alpha * weighted_difference[difference >= 0]
        loss_negative = (1 - self.alpha) * -weighted_difference[difference < 0]
        loss_sum = torch.sum(loss_positive) + torch.sum(loss_negative)
        return loss_sum


class TitledLoss(Module):
    def __init__(self, quantile) -> None:
        super(TitledLoss, self).__init__()
        self.quantile = quantile

    def forward(self, output: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        weights = weights / torch.sum(weights)
        e = (target - output)
        return torch.mean(weights * torch.maximum(self.quantile * e, (self.quantile - 1) * e))
