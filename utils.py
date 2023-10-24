""" Survey Data to diagnose and predict the risk of diabetes mellitus """

# Authors: Juan C. Vidal <juan.vidal@usc.es>
#          Marcos Matabuena <mmatabuena@hsph.harvard.edu>
#
# License: GNU General Public License, version 3 (or shortly GPL-3 license)

import torch
import numpy as np
import logging

from metrics import compute_metrics, compute_mean_metrics


def run_epoch(model, loader, criterion, optimizer, epoch=0, n_epochs=0,
              metric_type=0, train=True):

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
            metrics = compute_metrics(weight.detach().cpu().numpy(),
                                      output.detach().cpu().numpy(),
                                      target.detach().cpu().numpy(),
                                      metric_type)
            metrics_list.append(metrics)

        if train:
            logging.debug('Train: [Epoch {:04d}/{:04d}] [Batch {:04d}/{:04d}] Loss: {:.4f}'.format(
                epoch, n_epochs, batch + 1, len(loader), loss
            ))
        else:
            logging.debug('Eval : [Epoch ----/----] [Batch {:04d}/{:04d}] Loss: {:.4f}'.format(
                batch + 1, len(loader), loss
            ))

    return model, epoch_loss / len(loader) / loader.batch_size, \
        compute_mean_metrics(metrics_list, metric_type=metric_type)


def weighted_quantile(values, quantile, sample_weight):
    values = np.array(values).flatten()
    sample_weight = sample_weight.flatten()

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    weighted_quantiles = np.cumsum(sample_weight)
    index = np.argmax(weighted_quantiles >= min(quantile, 1))
    cutoff = values[index]

    return cutoff


