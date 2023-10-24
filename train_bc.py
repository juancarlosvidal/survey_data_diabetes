import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data import CustomDataset
from utils import run_epoch
from metrics import classification_mean_metrics
from models import BinaryClassificacionModel
from losses import CustomBCELoss
import logging


def cv_loop_bc(data, splits, n_epochs, batch_size, learning_rate, weight_decay,
               patience=5, min_delta=0):

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
            patience=patience,
            min_delta=min_delta)

        metrics_list.append(fold_test_metrics)
        if fold_min_loss < min_loss:
            model = fold_model
            min_loss = fold_min_loss

    return model, classification_mean_metrics(metrics_list)


# def run_fold(train_dl, val_dl, test_dl, n_epochs, learning_rate, weight_decay, best_error):
def run_fold(data, train_split, val_split, test_split, n_epochs, batch_size, learning_rate, weight_decay,
             patience=5, min_delta=0):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    min_loss = np.inf
    best_model = None

    input, target, weight= data['x'], data['y'], data['w']
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
    # model = LogisticRegression(5, 1)
    input_dim = train_x.shape[1]
    model = BinaryClassificacionModel(input_dim, input_dim, 1)
    model.to(device)

    # Declaring the criterion
    criterion = CustomBCELoss()

    # Declaring the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Declaring the scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    counter = 0
    for epoch in range(1, n_epochs + 1):
        # ####  TRAIN LOOP  #### #
        model, train_loss, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
            metric_type=0,
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
            metric_type=0,
            train=False
        )

        logging.debug('Epoch train loss {:.3f} val loss {:.3f}'.format(train_loss, val_loss))

        # Determine if model is the best
        if val_loss < min_loss:
            logging.debug('New min loss {:.5f}'.format(val_loss))
            min_loss = val_loss
            best_model = model
            # best_val_loader = val_loader
            counter = 0
        elif val_loss > (min_loss + min_delta):
            counter += 1
            logging.debug('Delta count {} and val_loss {:.3f}'.format(counter, val_loss))
            if counter >= patience:
                break

    # # Now we're going to wrap the model with a decorator that adds temperature scaling
    # model_t = ModelWithTemperature(best_model)
    # # Tune the model temperature, and save the results
    # model_t.set_temperature(best_val_loader)

    model_t = best_model

    # ####  TEST LOOP  #### #
    _, _, test_metrics = run_epoch(
        model=model_t,
        loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epoch=0,
        n_epochs=n_epochs,
        metric_type=0,
        train=False
    )

    logging.debug('Fold Test Acc: {:.3f} Prec: {:.3f} Rec: {:.3f} F1: {:.3f} CE: {:.3f} CM: {}'.format(
            test_metrics['accuracy'], test_metrics['precision'], test_metrics['recall'],
            test_metrics['f1'], test_metrics['ce'], test_metrics['cm']
        ))

    return model_t, min_loss, test_metrics

