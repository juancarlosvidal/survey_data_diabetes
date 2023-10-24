import os
import argparse
import pandas as pd
import random
import numpy as np

import torch
# torch.use_deterministic_algorithms(True)

from train import run_conformance_inference
from data import load_data

import logging
# logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s', level=logging.INFO)
logging.basicConfig(format='%(message)s', level=logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--input_dir', default="./input_real", help='Input_directory')
    parser.add_argument('-o', '--output_dir', default="./output", help='Output_directory')
    parser.add_argument('-c', '--combination', default=5, help='Selection of variables')
    parser.add_argument('-f', '--n_folds', default=5, help='Number of cv folders')
    parser.add_argument('-e', '--n_epochs', default=1000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', default=64, help='Batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.001, help='Learning rate')
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


    # Bioch: A1c, FPG, cholesterol, triglycerides
    # Metadata: age and gender
    # Clinical measures:  body mass index, waist circunference, diastolic and diastolic pressure, pulse, cholesterol
    comb_0 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBDSCHSI_43',
              'LBXSTR_43', 'LBXSGL_43', 'RIAGENDR', 'LBXGH_39']
    # Without A1c
    comb_1 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
              'BPXDI1', 'LBDSCHSI_43', 'LBXSTR_43', 'LBXSGL_43', 'RIAGENDR']
    # Without FPG
    comb_2 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBDSCHSI_43',
              'LBXSTR_43', 'RIAGENDR', 'LBXGH_39']
    # Same without A1c and FPG
    comb_3 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
              'BPXDI1', 'LBDSCHSI_43', 'LBXSTR_43', 'RIAGENDR']
    # Without any biochemical measures
    comb_4 = ['RIDAGEYR.x', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
              'BPXDI1', 'RIAGENDR']
    # Only metadata
    comb_5 = ['RIDAGEYR.x', 'RIAGENDR']
    # Only clinical measures
    comb_6 = ['BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS',
              'BPXDI1']

    # Selected variables
    all_combinations = [comb_0, comb_1, comb_2, comb_3, comb_4, comb_5, comb_6]
    selected_combination = all_combinations[int(args.combination)]
    output_var = 'diabetes'

    alphas = np.linspace(0.1, 1.0, num=10)
    dfi = pd.DataFrame()
    dfo = pd.DataFrame()
    onlyfiles = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    for f in onlyfiles:
        file = os.path.join(args.input_dir, f)
        logging.info('Running conformal inference for file {}'.format(file))

        df2 = pd.DataFrame()
        for combination in range(len(all_combinations)):
            selected_combination = all_combinations[combination]
            dfi = load_data(file, selected_combination, output_var)
            for alpha in alphas:
                ci_results, bc_metrics, correct = run_conformance_inference(
                    data=dfi,
                    n_folds=n_folds,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    alpha=alpha,
                    patience_classification=4,
                    min_delta_classification=0.0,
                    patience_regression=6,
                    min_delta_regression=0.0
                )

                # logging.info(ci_results)
                # logging.info(bc_metrics)

                df2['filename'] = np.repeat(os.path.basename(f), repeats=len(ci_results['condition_list']))
                df2['combination'] = np.repeat(combination, repeats=len(ci_results['condition_list']))
                df2['alpha'] = np.repeat(alpha, repeats=len(ci_results['condition_list']))
                df2['seqn'] = ci_results['seqn_list']
                df2['condition'] = np.round(ci_results['condition_list'], decimals=4)
                df2['quantile'] = np.round(ci_results['quantile_list'], decimals=4)
                df2['score'] = np.round(ci_results['score_list'], decimals=4)
                df2['class'] = ci_results['class_list']
                df2['y'] = ci_results['y_list']
                # df2['correct_score'] = np.round(ci_results['correct_list'], decimals=4)
                df2['correct_score'] = ci_results['correct_list']
                output_file = os.path.join(args.output_dir, 'out_' + '{:d}'.format(combination) +
                                           '_' + '{:.1f}'.format(alpha) +
                                           '_' + os.path.basename(f))
                df2.to_csv(output_file)

                result = {
                    'filename': os.path.basename(f),
                    'alpha': alpha,
                    'combination': combination,
                    'avg_correct_score': correct
                }

                output_df = pd.DataFrame([result])
                dfo = pd.concat([dfo, output_df], ignore_index=True)

    output_file = os.path.join(args.output_dir, "out_ci.csv")
    dfo.to_csv(output_file)
