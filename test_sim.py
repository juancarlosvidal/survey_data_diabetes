import os
import argparse
import pandas as pd
import random
import numpy as np

import torch
# torch.use_deterministic_algorithms(True)

from train import train_conformance_inference
from data_sim import load_data

import logging
# logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('-i', '--input_dir', default="./input_new_m", help='Input_directory')
    parser.add_argument('-i', '--input_dir', default="./conformance/sim", help='Input_directory')
    parser.add_argument('-o', '--output_dir', default="./output", help='Output_directory')
    parser.add_argument('-c', '--combination', default=0, help='Selection of variables')
    parser.add_argument('-f', '--n_folds', default=5, help='Number of cv folders')
    parser.add_argument('-e', '--n_epochs', default=1000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=64, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, help='Learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0.01, help='Weight decay')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    n_folds = args.n_folds
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    # alpha = 0.1

    # Case 1
    comb_0 = ['X1', 'X2', 'X3', 'X4', 'X5']
    # Case 1
    comb_1 = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']

    # Selected variables
    selected_combination = comb_0

    alphas = np.linspace(0.1, 1.0, num=10)
    dfi = pd.DataFrame()
    dfo = pd.DataFrame()
    onlyfiles = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    for f in onlyfiles:
        file = os.path.join(args.input_dir, f)
        logging.info('Running conformal inference for file {}'.format(file))

        data = load_data(file, selected_combination)

        n = len(data['y'])
        indices = random.sample(np.arange(n).tolist(), n)
        train_split = indices[:int((3*n) / 4)]
        test_split = indices[int((3*n) / 4):]

        train_data = {
            'x': data['x'][train_split],
            'y': data['y'][train_split],
            'w': data['w'][train_split],
            'z': data['z'][train_split]
        }

        # Scale training data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_data['x'] = scaler.fit_transform(train_data['x'])

        # Upsample unbalanced data of the training set
        from sklearn.utils import resample

        y = train_data['y'].reshape(-1)
        c1 = np.argwhere(y == 1.0).reshape(-1)
        c2 = np.argwhere(y == 0.0).reshape(-1)
        print('C1: {}'.format(c1.shape))
        print('C2: {}'.format(c2.shape))

        if c1.shape[0] < c2.shape[0]:
            c1_upsample = resample(c1, replace=True, n_samples=len(c2), random_state=42)
            upsampled = np.concatenate([c1_upsample, c2]).reshape(-1)
        else:
            c2_upsample = resample(c2, replace=True, n_samples=len(c1), random_state=42)
            upsampled = np.concatenate([c1, c2_upsample]).reshape(-1)
        print('Upsampled dataset: {}'.format(upsampled.shape))

        train_data = {
            'x': data['x'][upsampled],
            'y': data['y'][upsampled],
            'w': data['w'][upsampled],
            'z': data['z'][upsampled]
        }

        test_data = {
            'x': data['x'][test_split],
            'y': data['y'][test_split],
            'w': data['w'][test_split],
            # 't': data['t'][test_split],
            'z': data['z'][test_split]
        }

        # Scale test data
        test_data['x'] = scaler.transform(test_data['x'])

        for alpha in alphas:

            predictor = train_conformance_inference(
                data=train_data,
                n_folds=n_folds,
                n_epochs=n_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                alpha=alpha,
                patience_classification=4,
                min_delta_classification=0.0,
                patience_regression=6,
                min_delta_regression=0.0,
                hidden_sizes_bc=[64],
                hidden_sizes_rr=[64]
            )

            ci_result, correct = predictor.classify(test_data)

            result = {
                'filename': os.path.basename(f),
                'alpha': alpha,
                'correct': correct
            }

            output_df = pd.DataFrame([result])
            dfo = pd.concat([dfo, output_df], ignore_index=True)

    print("Saving file: {}".format(os.path.join(args.output_dir, "out_ci.csv")))
    output_file = os.path.join(args.output_dir, os.path.basename(args.input_dir) + '_' + 'out_ci.csv')
    dfo.to_csv(output_file)
