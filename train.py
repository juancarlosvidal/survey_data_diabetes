import random
import logging

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import KFold

from train_bc import cv_loop_bc
from train_rr import cv_loop_rr
from utils import weighted_quantile


def run_conformance_inference(data, n_folds, n_epochs, batch_size, learning_rate, weight_decay,
                              alpha, patience_classification=5, min_delta_classification=0,
                              patience_regression=5, min_delta_regression=0):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n = len(data['y'])
    indices = random.sample(np.arange(n).tolist(), n)

    set1 = indices[:int(n/4)]
    set2 = indices[int(n/4):int(2*n/4)]
    set3 = indices[int(2*n/4):int(3*n/4)]
    set4 = indices[int(3*n/4):]
    # sigma = data['sigma']

    data1 = {
        'x': data['x'][set1],
        'y': data['y'][set1],
        'w': data['w'][set1],
        'z': data['z'][set1]
    }

    k_fold = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    indexes = sorted(range(len(data1['w'])-1))
    splits = k_fold.split(indexes)

    model_ce, metrics_ce = cv_loop_bc(
        data=data1,
        splits=splits,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience_classification,
        min_delta=min_delta_classification)

    data2 = {
        'x': data['x'][set2],
        'y': model_ce(torch.from_numpy(data['x'][set2]).to(device)).cpu().detach().numpy(),
        # 'y': torch.log(model_ce(torch.from_numpy(data['x'][set2])).to(device)).cpu().detach().numpy(),
        'w': data['w'][set2],
        'z': data['z'][set2]
    }

    etiqueta = data['y'][set2]
    donde = np.where(etiqueta == 0)
    data2['y'][donde] = (1 - data2['y'][donde])

    # data2['y'] = data2['y'] + data2['z'] * sigma

    k_fold = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    indexes = sorted(range(len(data2['w'])-1))
    splits = k_fold.split(indexes)

    # from sklearn.utils.fixes import sp_version, parse_version
    # solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    # from sklearn.linear_model import QuantileRegressor
    # qr = QuantileRegressor(quantile=alpha, alpha=0, solver=solver)
    # qrf = qr.fit(data2['x'], data2['y'])

    model_rr, metrics_rr = cv_loop_rr(
        data=data2,
        splits=splits,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        alpha=alpha,
        patience=patience_regression,
        min_delta=min_delta_regression)

    data3 = {
        'x': data['x'][set3],
        'y': model_rr(torch.from_numpy(data['x'][set3]).to(device)).cpu().detach().numpy(),
        # 'y': torch.exp(model_rr(torch.from_numpy(data['x'][set3])).to(device)).cpu().detach().numpy(),
        # 'y': qrf.predict(data['x'][set3]).reshape(-1, 1),
        'w': data['w'][set3],
        'z': data['z'][set3]
    }

    etiqueta = data['y'][set3]
    donde = np.where(etiqueta == 0)
    data3['y'][donde] = (1 - data3['y'][donde])

    # data3['y'][np.where(data3['y'] < 0)] = 0
    # data3['y'][np.where(data3['y'] > 1)] = 1

    # Compute conformity scores on data3 set
    empirical_quantile = (1 - alpha) * (1 + 1 / data3['x'].shape[0])
    logging.debug("Empirical quantile 1: {}".format(empirical_quantile))

    # score_output = model_ce(torch.from_numpy(data3['x']).to(device)).cpu().detach().numpy() + data3['z'] * sigma
    score_output = model_ce(torch.from_numpy(data3['x']).to(device)).cpu().detach().numpy()
    quant_output = data3['y']
    logging.debug('Score output shape {}'.format(score_output.shape))
    logging.debug('Quantile output shape {}'.format(quant_output.shape))

    negative_index = np.where(data['y'][set3] == 0)
    score_output[negative_index] = (1 - score_output[negative_index])
    conformity_scores_3 = quant_output - score_output

    data4 = {
        'x': data['x'][set4],
        'y': model_rr(torch.from_numpy(data['x'][set4]).to(device)).cpu().detach().numpy(),
        # 'y': torch.exp(model_rr(torch.from_numpy(data['x'][set4])).to(device)).cpu().detach().numpy(),
        # 'y': qrf.predict(data['x'][set4]).reshape(-1, 1),
        'w': data['w'][set4],
        'z': data['z'][set4]
    }

    etiqueta = data['y'][set4]
    donde = np.where(etiqueta == 0)
    data4['y'][donde] = (1 - data4['y'][donde])

    # data4['y'][np.where(data4['y'] < 0)] = 0
    # data4['y'][np.where(data4['y'] > 1)] = 1

    tw_4 = data4['w'] / np.sum(data4['w'])
    y = data['y'][set4]
    # y = data['y'][set4] + data4['z'] * sigma
    # y = data4['y'] + data4['z'] * sigma
    real_list = []
    patient_class = []
    condition_list = []
    correct_list = []
    correct_1, correct_2, correct_3, correct_4 = 0, 0, 0, 0
    empirical_quantile_4 = (1 - alpha) * (1 + 1 / data4['x'].shape[0])
    score_output_4 = model_ce(torch.from_numpy(data4['x']).to(device)).cpu().detach().numpy()
    quant_output_4 = data4['y']
    # quant_output_4 = torch.exp(data4['y']).cpu().detach().numpy()

    for i in range(len(data4['x'])):
        tw_aux = np.append(data3['w'], data4['w'][i])
        tw_aux = tw_aux / np.sum(tw_aux)
        conformity_scores_aux = np.append(conformity_scores_3, 1e20)
        empirical_quantile_aux = weighted_quantile(conformity_scores_aux, empirical_quantile_4, tw_aux)
        # empirical_quantile_aux = np.quantile(conformity_scores_aux, empirical_quantile_4)

        condition = quant_output_4[i] - empirical_quantile_aux
        condition_list.append(condition.item())

        score_pos = score_output_4[i]
        score_neg = 1 - score_output_4[i]

        if condition <= score_pos and condition <= score_neg:
            patient_class.append(2)
        elif condition <= score_pos:
            patient_class.append(1)
        elif condition <= score_neg:
            patient_class.append(0)
        else:
            patient_class.append(-1)

        if patient_class[i] == 2:
            correct_2 += 1 * float(tw_4[i])
            correct_list.append(1 * float(tw_4[i]))
        else:
            correct_2 += (patient_class[i] == int(y[i])) * float(tw_4[i])
            correct_list.append((patient_class[i] == int(y[i])))

        correct_4 += (patient_class[i] == int(y[i])) * float(tw_4[i])

        logging.info("Alpha: {:.3f} - Condition: {:.3f} - P1: {:.3f} - P0: {:.3f} - PC/YC: {:2d}/{:2d} ({:5})"
                     .format(alpha, condition.item(), score_pos.item(), score_neg.item(),
                             patient_class[i], int(y[i].item()), str(patient_class[i] == int(y[i]))))

    # logging.info("Prediction_class vs probability_class: {:.3f}".format(correct_1))
    logging.info("Prediction_class vs y_class: {:.3f}".format(correct_2))
    # logging.info("Strict prediction_class vs probability_class: {:.3f}".format(correct_3))
    logging.info("Strict prediction_class vs y_class: {:.3f}".format(correct_4))

    ci_result = {
        'seqn_list': data['seqn'][set4],
        'condition_list': condition_list,
        'quantile_list': quant_output_4,
        'score_list': score_output_4,
        'class_list': patient_class,
        'y_list': data['y'][set4],
        'correct_list': correct_list
    }

    return ci_result, metrics_ce, correct_2
