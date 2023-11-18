""" Survey Data to diagnose and predict the risk of diabetes mellitus """

# Authors: Juan C. Vidal <juan.vidal@usc.es>
#          Marcos Matabuena <mmatabuena@hsph.harvard.edu>
#
# License: GNU General Public License, version 3 (or shortly GPL-3 license)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data import CustomDataset
from utils import run_epoch
from metrics import regression_mean_metrics
from models import RegressionModel
from losses import PinballLoss
import logging


def cv_loop_rr(data, splits, n_epochs, batch_size, learning_rate, weight_decay,
               alpha=0.1, patience=5, min_delta=0):
    """
    cv_loop_rr (Cross-Validation Loop for Regression) is a function designed to perform K-Fold cross-validation for a regression task. It involves training a regression model on each fold, evaluating it on a validation set, and testing it on a separate test set. The function aims to identify the model with the best performance across folds and aggregate the test metrics.

    :param data (dict): 
        A dictionary containing features (x), labels (y), and possibly other data elements like weights.
    :param splits (generator): 
        A generator of train-test splits, typically from K-Fold cross-validation.
    :param n_epochs (int): 
        Number of epochs for training the model.
    :param batch_size (int): 
        Batch size used during model training.
    :param learning_rate (float): 
        Learning rate for the optimizer.
    :param weight_decay (float): 
        Weight decay parameter for regularization.
    :param alpha (float, optional): 
        Quantile level used in regression, relevant for quantile regression tasks. Default is 0.1.
    :param patience (int, optional): 
        Patience parameter for early stopping to prevent overfitting. Default is 5.
    :param min_delta (float, optional): 
        Minimum change in validation loss required to qualify as an improvement. Default is 0.
    :returns:
        - model (model object): The best-performing model across all folds, as determined by the minimum loss on the validation set.
        - mean_metrics (dict): A dictionary containing the mean of the test metrics across all folds. Computed by the regression_mean_metrics function.
    """
    model = None
    min_loss = np.inf
    metrics_list = []
    for train_split, test_split in splits:
        # Select the 20% of the train size as the validation set
        train_size = round(len(train_split) * 0.8)
        train_index, val_index, test_index = train_split[:train_size], train_split[train_size:], test_split

        fold_model, fold_min_loss, fold_test_metrics = run_fold(
            data=data,
            train_split=train_index,
            val_split=val_index,
            test_split=test_index,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            alpha=alpha,
            patience=patience,
            min_delta=min_delta)

        metrics_list.append(fold_test_metrics)
        if fold_min_loss < min_loss:
            model = fold_model
            min_loss = fold_min_loss

    return model, regression_mean_metrics(metrics_list)


def run_fold(data, train_split, val_split, test_split, n_epochs, batch_size, learning_rate, weight_decay,
             alpha=0.1, patience=5, min_delta=0):
    """
    run_fold is a function tailored for training, validating, and testing a regression model on a single fold of data. This function is essential in a cross-validation process, handling the entire lifecycle of model training, including data preprocessing, training loops, early stopping, and evaluation on test data.

    :param data (dict): 
        A dictionary containing features (x), labels (y), and weights (w).
    :param train_split (array-like): 
        Indices for training samples.
    :param val_split (array-like): 
        Indices for validation samples.
    :param test_split (array-like):
        Indices for test samples.
    :param n_epochs (int): 
        Number of epochs for training the model.
    :param batch_size (int): 
        Batch size used during model training.
    :param learning_rate (float): 
        Learning rate for the optimizer.
    :param weight_decay (float):    
        Weight decay parameter for regularization.
    :param alpha (float, optional): 
        Quantile level for the Pinball Loss, relevant in quantile regression. Default is 0.1.
    :param patience (int, optional): 
        Patience parameter for early stopping. Default is 5.
    :param min_delta (float, optional): 
        Minimum change in validation loss required to qualify as an improvement. Default is 0.
    :returns:
        - model_t (model object): The best-performing model on the validation set.
        - min_loss (float): The minimum loss achieved on the validation set.
        - test_metrics (dict): A dictionary containing test metrics such as mean absolute error (MAE), mean squared error (MSE), and R-squared (R2).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    min_loss = np.inf
    best_model = None

    input, target, weight = data['x'], data['y'], data['w']

    scaler = StandardScaler()
    train_x = scaler.fit_transform(input[train_split])
    val_x = scaler.fit_transform(input[val_split])
    test_x = scaler.fit_transform(input[test_split])

    train_input, train_target, train_weight = \
        train_x, target[train_split], weight[train_split]
    val_input, val_target, val_weight = \
        val_x, target[val_split], weight[val_split]
    test_input, test_target, test_weight = \
        test_x, target[test_split], weight[test_split]

    train_loader, val_loader, test_loader = \
        DataLoader(CustomDataset(train_input, train_target, train_weight), batch_size=batch_size), \
        DataLoader(CustomDataset(val_input, val_target, val_weight), batch_size=batch_size), \
        DataLoader(CustomDataset(test_input, test_target, test_weight), batch_size=batch_size)

    # Declaring the model
    input_dim = train_input.shape[1]
    model = RegressionModel(input_dim, 64, 1)
    model.to(device)

    # Declaring Cthe criterion
    criterion = PinballLoss(alpha)

    # Declaring the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Declaring the scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    counter = 0
    for epoch in range(1, n_epochs + 1):
        # ####  TRAIN LOOP  #### #
        _, train_loss, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            metric_type=1,
            train=True
        )

        scheduler.step()

        # ####  VALIDATION LOOP  #### #
        _, val_loss, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            metric_type=1,
            train=False
        )

        logging.debug('Epoch train loss {:.3f} val loss {:.3f}'.format(train_loss, val_loss))

        # Determine if model is the best
        if val_loss < min_loss:
            logging.debug('New min loss {:.3f}'.format(val_loss))
            min_loss = val_loss
            best_model = model
            counter = 0
        elif val_loss > (min_loss + min_delta):
            counter += 1
            logging.debug('Delta count {} and val_loss {:.3f}'.format(counter, val_loss))
            if counter >= patience:
                break

    model_t = best_model

    # ####  TEST LOOP  #### #
    _, _, test_metrics = run_epoch(
        model=model_t,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        n_epochs=n_epochs,
        metric_type=1,
        train=False
    )

    logging.debug('Fold Test MAE: {:.3f} MSE: {:.3f} R2: {:.3f}'.format(
        test_metrics['mae'], test_metrics['mse'], test_metrics['r2']
    ))

    return model_t, min_loss, test_metrics

