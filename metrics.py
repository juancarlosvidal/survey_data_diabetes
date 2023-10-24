import copy
import numpy as np
from torchmetrics.classification import BinaryAccuracy, MultilabelAccuracy
from torchmetrics import MeanSquaredError
from sklearn.metrics import roc_auc_score


def weighted_binary_accuracy(yp, yt, w):
    weights = w / np.sum(w)
    yp_tag = np.round(yp)
    correct_results = (yp_tag == yt)
    correct_results = weights * correct_results.type(np.FloatTensor)
    correct_results = correct_results.sum().float()
    return correct_results


def binary_accuracy(yp, yt):
    accuracy = BinaryAccuracy()
    return accuracy(yp, yt)


def multiclass_accuracy(yp, yt, w, n_classes=2):
    accuracy = MultilabelAccuracy(task="multiclass", num_labels=n_classes)
    return accuracy(yp, yt)


def weighted_mse_accuracy(yp, yt, w):
    weights = w / np.sum(w)
    return np.sum(weights * np.pow(yp - yt, 2), axis=0)


def mse_accuracy(yp, yt, w):
    accuracy = MeanSquaredError()
    return accuracy(yp, yt)


def weighted_precision(yp, yt, w):
    weights = w / np.sum(w)
    return np.sum(weights * np.pow(yp - yt, 2), axis=0)


def compute_metrics(weights, output, target, metric_type=0):
    if metric_type == 0:
        return compute_classification_metrics(weights, output, target)
    else:
        return compute_regression_metrics(weights, output, target)


def compute_classification_metrics(weights, output, target):
    if len(target) > 1:
        output = np.squeeze(np.asarray(output))
        weights = np.squeeze(np.asarray(weights))
        target = np.squeeze(copy.copy(target))
    pred = np.round(output)

    auc_score = auc(weights, output, target)
    confusion_matrix = cm(pred, target)

    acc = accuracy(weights, pred, target)
    prec = precision(weights, pred, target)
    rec = recall(weights, pred, target)
    f1_score = f1(weights, pred, target)
    ce = cross_entropy(weights, pred, target)
    # ce = cross_entropy(weights, output, target)

    metrics = {
        'auc': auc_score,
        'accuracy': acc.item(),
        'recall': rec.item(),
        'precision': prec.item(),
        'f1': f1_score.item(),
        'cm': confusion_matrix.tolist(),
        'ce': ce.item()
    }

    return metrics


def auc(weights, output, target):
    unique_target = np.unique(target)
    all_auc = 0
    # output_classes = [np.round(output[i], decimals=2) for i in range(unique_target.shape[0])]
    output_classes = [(1-output), output]
    class_indices = [[i for i in range(len(target)) if target[i] == label] for label in unique_target]
    rest_indices = [[i for i in range(len(target)) if target[i] != label] for label in unique_target]

    for l, label in enumerate(unique_target):
        suma = 0
        auc = 0
        for i in class_indices[l]:
            for j in rest_indices[l]:
                a = weights[i] * weights[j]

                auc += (a * int(output_classes[l][i] < output_classes[l][j])) + \
                       (0.5 * a * int(output_classes[l][i] == output_classes[l][j]))
                suma += a
        all_auc += 1 - (auc / suma)

    return all_auc / unique_target.shape[0]
    # value = 0
    # if len(target) >= 5:
    #     value = roc_auc_score(target, output)
    # return value


def cm(output, target, classes=2):
    cma = np.zeros((classes, classes))

    # nt = target.numpy()
    for i in range(len(target)):
        # a = np.argmax(output[i])
        a = int(output[i])
        b = int(target[i])
        cma[b, a] += 1

    return cma


def accuracy(weights, pred, target):
    correct_pred = (pred == target)
    acc = np.sum(correct_pred * weights) / np.sum(weights)
    return acc


def recall(weights, pred, target):
    if len(np.unique(target)) > 2:
        # Macro averaging
        total_recall = 0
        for i in np.unique(target):
            true_positive = ((target == np.ones(target.shape[0]) * i) & (pred == np.ones(target.shape[0]) * i))
            false_negative = ((target == np.ones(target.shape[0]) * i) & (pred != np.ones(target.shape[0]) * i))
            recall = np.sum(true_positive * weights) / \
                     (np.sum((true_positive + false_negative) * weights) + np.finfo(float).eps)
            total_recall += recall
        return total_recall / np.unique(target).shape[0]
    else:
        true_positive = ((target == np.ones(target.shape[0])) & (pred == np.ones(target.shape[0])))
        false_negative = ((target == np.ones(target.shape[0])) & (pred == np.zeros(target.shape[0])))
        recall = np.sum(true_positive * weights) / \
                 (np.sum((true_positive + false_negative) * weights) + np.finfo(float).eps)
        return recall


def precision(weights, pred, target):
    if len(np.unique(target)) > 2:
        # Macro averaging
        total_precision = 0
        for i in np.unique(target):
            true_positive = ((target == np.ones(target.shape[0]) * i) & (pred == np.ones(target.shape[0]) * i))
            false_positive = ((target != np.ones(target.shape[0]) * i) & (pred == np.ones(target.shape[0]) * i))
            precision = np.sum(true_positive * weights) / \
                        (np.sum((true_positive + false_positive) * weights) + np.finfo(float).eps)
            total_precision += precision
        return total_precision / np.unique(target).shape[0]
    else:
        true_positive = ((target == np.ones(target.shape[0])) & (pred == np.ones(target.shape[0])))
        false_positive = ((target == np.zeros(target.shape[0])) & (pred == np.ones(target.shape[0])))
        precision = np.sum(true_positive * weights) / \
                    (np.sum((true_positive + false_positive) * weights) + np.finfo(float).eps)
        return precision


def f1(weights, pred, target):
    pr = precision(weights, pred, target)
    rc = recall(weights, pred, target)
    f1 = 2 * pr * rc / (pr + rc+  np.finfo(float).eps)
    return f1


def cross_entropy(weights, output, target):
    weights = weights / np.sum(weights)
    yp = np.clip(output, 1e-7, 1 - 1e-7)
    term_1 = target * np.log(yp)
    return -np.sum(weights * term_1)


def compute_regression_metrics(weights, pred, labels):
    weights = np.squeeze(weights)
    dif = np.squeeze(np.asarray(np.abs(pred - labels)))
    weighted_dif = np.multiply(weights, dif)
    mae = np.sum(weighted_dif) / np.sum(weights)

    sq_dif = np.power(dif, 2)
    weighted_sq_dif = np.multiply(weights, sq_dif)
    mse = np.sum(weighted_sq_dif) / np.sum(weights)

    norm_weights = weights / np.sum(weights)
    a = np.sum(np.multiply(np.power(dif, 2), norm_weights))
    b1 = np.squeeze(labels)
    b2 = np.dot(np.squeeze(labels), norm_weights)
    b = np.sum(np.multiply(np.power(np.subtract(b1, b2), 2), norm_weights))
    r2 = 1 - (a / b)
    metrics = {
        'mae': mae,
        'mse': mse,
        'r2': r2
    }

    return metrics


def compute_mean_metrics(list, classes=2, metric_type=0):
    if metric_type == 0:
        return classification_mean_metrics(list, classes)
    else:
        return regression_mean_metrics(list)


def classification_mean_metrics(dict_list, classes=2):
    mean_dict = {}
    for key in dict_list[0].keys():
        if key == 'cm':
            x = np.zeros((classes, classes))
            for d in dict_list:
                for i in range(classes):
                    for j in range(classes):
                        x[i,j] = x[i,j] + d[key][i][j]
            mean_dict[key] = x.tolist()
        else:
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def regression_mean_metrics(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
