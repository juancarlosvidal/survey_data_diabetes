""" Survey Data to diagnose and predict the risk of diabetes mellitus """

# Authors: Juan C. Vidal <juan.vidal@usc.es>
#          Marcos Matabuena <mmatabuena@hsph.harvard.edu>
#
# License: GNU General Public License, version 3 (or shortly GPL-3 license)

import torch
import numpy as np
import logging

from metrics import compute_metrics, compute_mean_metrics


def run_epoch(
    model, loader, criterion, optimizer, epoch=0, n_epochs=0, metric_type=0, train=True
):
    """
    run_epoch is a versatile function designed to execute a single training or
    evaluation epoch for a given model using a specified data loader. It handles
    both the training and evaluation phases, integrates custom loss calculation
    with optional sample weights, and calculates metrics based on the model's
    output and targets.

    :param model (nn.Module):
        The neural network model to be trained or evaluated.
    :param loader (DataLoader):
        A PyTorch DataLoader that provides batches of data (inputs, targets, and
        optional weights).
    :param criterion (function):
        The loss function used for training or evaluation. It can be a standard
        loss function like BCELoss or a custom one that handles weights.
    :param optimizer (Optimizer):
        The optimizer used for updating the model parameters during training.
    :param epoch (int, optional):
        The current epoch number. Default is 0.
    :param n_epochs (int, optional):
        The total number of epochs for training. Default is 0.
    :param metric_type (int, optional):
        An identifier to specify the type of metrics to compute. Default is 0.
    :param train (bool, optional):
        A flag indicating whether the function is being used for training (True)
        or evaluation (False). Default is True.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metrics = None

    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    metrics_list = []
    for batch, (input, target, weight) in enumerate(loader):
        # Transfer Data to GPU if available
        input, target, weight = input.to(device), target.to(device), weight.to(device)
        if train:
            # Setting our stored gradients equal to zero
            optimizer.zero_grad()

            # Forward Pass
            output = model(input)
            # Get the Loss
            # loss = criterion(output, target)  # nn.BCELoss()
            loss = criterion(output, target, weight)  # CustomLoss()
            # loss = criterion(pred, torch.squeeze(y))  # nn.CrossEntropyLoss()

            # Computes the gradient of the given tensor w.r.t. the weights/bias
            loss.backward()
            # Update weight
            optimizer.step()
            # Update train_loss

        else:
            with torch.no_grad():
                output = model(input)
                # loss = criterion(output, target)  # nn.BCELoss()
                loss = criterion(output, target, weight)  # CustomLoss()

        epoch_loss += loss.item()

        # Accounting
        # _, predictions = torch.topk(output, 1)
        # error = 1 - torch.eq(predictions, target).float().mean()
        # accuracy = 1 - weighted_binary_accuracy(output, target, weight).item()
        # accuracy = 1 - binary_accuracy(output, target).item()

        if len(target) == loader.batch_size:
            metrics = compute_metrics(
                weight.detach().cpu().numpy(),
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                metric_type,
            )
            metrics_list.append(metrics)

        if train:
            logging.debug(
                "Train: [Epoch {:04d}/{:04d}] [Batch {:04d}/{:04d}] Loss: {:.4f}".format(
                    epoch, n_epochs, batch + 1, len(loader), loss
                )
            )
        else:
            logging.debug(
                "Eval : [Epoch ----/----] [Batch {:04d}/{:04d}] Loss: {:.4f}".format(
                    batch + 1, len(loader), loss
                )
            )

    return (
        model,
        epoch_loss / len(loader) / loader.batch_size,
        compute_mean_metrics(metrics_list, metric_type=metric_type),
    )


def weighted_quantile(values, quantile, sample_weight):
    """
    weighted_quantile is a function designed to calculate the quantile of a given
    set of values, taking into account the weights associated with each value.
    This is particularly useful in statistics and data analysis where the
    distribution of values is uneven, and some values carry more weight or
    importance than others.

    :param values (array-like):
        An array or list of numerical values for which the quantile is to be
        calculated.
    :param quantile (float):
        The target quantile to calculate, a value between 0 and 1, where 0.5
        corresponds to the median.
    :param sample_weight (array-like):
        An array or list of weights corresponding to each value in values. Each
        weight indicates the importance or frequency of the corresponding value.
    """
    values = np.array(values).flatten()
    sample_weight = sample_weight.flatten()

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    weighted_quantiles = np.cumsum(sample_weight)
    index = np.argmax(weighted_quantiles >= min(quantile, 1))
    cutoff = values[index]

    return cutoff
