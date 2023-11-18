""" Survey Data to diagnose and predict the risk of diabetes mellitus """

# Authors: Juan C. Vidal <juan.vidal@usc.es>
#          Marcos Matabuena <mmatabuena@hsph.harvard.edu>
#
# License: GNU General Public License, version 3 (or shortly GPL-3 license)

import copy
import numpy as np
from torchmetrics.classification import BinaryAccuracy, MultilabelAccuracy
from torchmetrics import MeanSquaredError
from sklearn.metrics import roc_auc_score


def weighted_binary_accuracy(yp, yt, w):
    """
    weighted_binary_accuracy is a function designed to calculate the accuracy
    of binary classification predictions, taking into account the weights
    associated with each prediction. This approach is useful in datasets where
    certain samples are more significant than others, and a simple unweighted
    accuracy would not fully capture the model's performance.

    :param yp (array-like):
        Predicted probabilities or binary predictions from the model. Typically,
        these are values between 0 and 1.
    :param yt (array-like):
        Ground truth binary labels corresponding to the predictions. These are
        usually 0 or 1.
    :param w (array-like):
        Weights associated with each prediction, indicating the importance or
        significance of each sample.
    :returns:
        correct_results (float): The weighted accuracy of the binary predictions,
        a value between 0 and 1, where 1 indicates perfect accuracy.
    """
    weights = w / np.sum(w)
    yp_tag = np.round(yp)
    correct_results = yp_tag == yt
    correct_results = weights * correct_results.type(np.FloatTensor)
    correct_results = correct_results.sum().float()
    return correct_results


def binary_accuracy(yp, yt):
    """
    binary_accuracy is a function designed to calculate the accuracy of binary
    classification predictions. This function utilizes a BinaryAccuracy class or
    metric, typically available in deep learning frameworks, to compute the accuracy.
    The accuracy is determined by comparing the predicted values against the true
    labels.

    :param yp (array-like or Tensor):
        Predicted probabilities or binary predictions from the model. These can be
        values between 0 and 1 (probabilities) or directly binary values (0 or 1).
    :param yt (array-like or Tensor):
        Ground truth binary labels corresponding to the predictions. These are
        usually 0 or 1.
    :returns:
        The accuracy score computed by the BinaryAccuracy metric. This score is a
        float value between 0 and 1, where 1 indicates perfect accuracy.
    """
    accuracy = BinaryAccuracy()
    return accuracy(yp, yt)


def multiclass_accuracy(yp, yt, w, n_classes=2):
    """
    multiclass_accuracy is a function designed to calculate the accuracy of
    multiclass classification predictions. This function leverages a MultilabelAccuracy
    class or metric, which is typically available in various deep learning frameworks,
    to compute the accuracy for a multiclass classification task. The function is
    designed to handle scenarios with more than two classes.

    :param yp (array-like or Tensor):
        Predicted probabilities or labels from the model. For multiclass classification,
        these predictions are usually in the form of probabilities for each class.
    :param yt (array-like or Tensor):
        Ground truth labels corresponding to the predictions, typically represented as
        integers denoting class indices.
    :param w (array-like or Tensor):
        Weights associated with each prediction. However, it's important to note that
        in the current implementation, these weights are not used.
    :param n_classes (int, optional):
        The number of classes in the multiclass classification task. Default value is 2.
    :returns:
        The accuracy score computed by the MultilabelAccuracy metric. This score is a
        float value between 0 and 1, where 1 indicates perfect accuracy.
    """
    accuracy = MultilabelAccuracy(task="multiclass", num_labels=n_classes)
    return accuracy(yp, yt)


def weighted_mse_accuracy(yp, yt, w):
    """
    weighted_mse_accuracy is a function designed to calculate the weighted Mean
    Squared Error (MSE) for a set of predictions, particularly in regression tasks.
    This function takes into account the weights associated with each prediction,
    allowing for a more nuanced evaluation in datasets where certain samples have
    more significance than others.

    :param yp (array-like):
        Predicted values from the model. These can be continuous values corresponding
        to the regression task.
    :param yt (array-like):
        Ground truth values corresponding to the predictions. These are also continuous
        values.
    :param w (array-like):
        Weights associated with each prediction, indicating the importance or
        significance of each sample.
    :returns:
        The function returns the total weighted MSE, a float value representing
        the average of the squared differences between the predictions and the
        actual values, weighted by their importance.
    """
    weights = w / np.sum(w)
    return np.sum(weights * np.pow(yp - yt, 2), axis=0)


def mse_accuracy(yp, yt, w):
    """
    mse_accuracy is a function designed to calculate the Mean Squared Error (MSE)
    for a set of predictions, typically used in regression tasks. This function
    utilizes a MeanSquaredError class or metric, often available in deep learning
    frameworks, to compute the MSE. The accuracy in this context refers to the
    closeness of the predictions to the actual values, measured by the average of
    squared differences.

    :param yp (array-like or Tensor):
        Predicted values from the model. These are continuous values corresponding
        to the regression task.
    :param yt (array-like or Tensor):
        Ground truth values corresponding to the predictions. These are also
        continuous values.
    :param w (array-like or Tensor):
        Weights associated with each prediction. However, it's important to note
        that in the current implementation, these weights are not used.
    :returns:
        The function the MSE computed by the MeanSquaredError metric. This value is
        a float representing the average squared difference between the predictions
        and the actual values.
    """
    accuracy = MeanSquaredError()
    return accuracy(yp, yt)


def compute_metrics(weights, output, target, metric_type=0):
    """
    compute_metrics is a versatile function designed to compute either classification
    or regression metrics based on a specified metric type. It acts as a wrapper that
    delegates the computation to either compute_classification_metrics or
    compute_regression_metrics, depending on the context indicated by the metric_type
    parameter. This design makes the function adaptable for use in different types of
    machine learning tasks.

    :param weights (array-like):
        An array of weights associated with each data point, indicating the importance
        or significance of each sample in the computation of metrics.
    :param output (array-like):
        The predictions made by the model. For classification tasks, these might be
        probabilities or class labels; for regression tasks, these are continuous values.
    :param target (array-like):
        The ground truth values or labels corresponding to the predictions.
    :param metric_type (int, optional):
        An integer that specifies the type of metrics to compute. A typical convention
        might be 0 for classification metrics and 1 (or any non-zero value) for
        regression metrics. Default value is 0.
    :returns:
        The result of the metrics computation, which depends on the specific
        implementation of compute_classification_metrics or compute_regression_metrics.
        This could be a single metric value, a dictionary of multiple metrics, or any other format suitable for representing the performance of the model in the given task.
    """
    if metric_type == 0:
        return compute_classification_metrics(weights, output, target)
    else:
        return compute_regression_metrics(weights, output, target)


def compute_classification_metrics(weights, output, target):
    """
    compute_classification_metrics is a function designed to compute a suite of
    metrics relevant to classification tasks. It takes into account both the
    predicted and true labels, as well as weights associated with each sample.
    This function calculates various metrics including AUC, accuracy, precision,
    recall, F1 score, confusion matrix, and cross-entropy.

    :param weights (array-like):
        Weights associated with each sample, emphasizing the importance or influence
        of individual samples in metric computation.
    :param output (array-like):
        Predicted values or probabilities from the model. These are typically
        continuous values that may need to be thresholded or rounded to obtain
        binary predictions.
    :param target (array-like):
        Ground truth binary labels corresponding to the predictions.
    :returns:
        metrics (dict): A dictionary containing the following key-value pairs
        representing different classification metrics:
            - auc: AUC score.
            - accuracy: Accuracy score.
            - recall: Recall score.
            - precision: Precision score.
            - f1: F1 score.
            - cm: Confusion matrix.
            - ce: Cross-entropy score.
    """
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
        "auc": auc_score,
        "accuracy": acc.item(),
        "recall": rec.item(),
        "precision": prec.item(),
        "f1": f1_score.item(),
        "cm": confusion_matrix.tolist(),
        "ce": ce.item(),
    }

    return metrics


def auc(weights, output, target):
    """
    auc is a custom function designed to compute the Area Under the Curve (AUC)
    for binary classification tasks, with an extension to handle multi-class
    scenarios by averaging the AUCs for each class. This function calculates AUC
    in a pairwise manner, considering the weights of each sample in the dataset,
    which makes it particularly suitable for datasets where different samples
    have varying levels of importance.

    :param weights (array-like):
        Weights associated with each sample in the dataset.
    :param output (array-like):
        Predicted probabilities from the model for each class. These are continuous
        values typically between 0 and 1.
    :param target (array-like):
        Ground truth labels corresponding to the predictions.
    :returns:
        The function returns the average AUC score over all classes, a float value
        representing the average performance of the model across different classes.
    """
    unique_target = np.unique(target)
    all_auc = 0
    # output_classes = [np.round(output[i], decimals=2) for i in range(unique_target.shape[0])]
    output_classes = [(1 - output), output]
    class_indices = [
        [i for i in range(len(target)) if target[i] == label] for label in unique_target
    ]
    rest_indices = [
        [i for i in range(len(target)) if target[i] != label] for label in unique_target
    ]

    for l, label in enumerate(unique_target):
        suma = 0
        auc = 0
        for i in class_indices[l]:
            for j in rest_indices[l]:
                a = weights[i] * weights[j]

                auc += (a * int(output_classes[l][i] < output_classes[l][j])) + (
                    0.5 * a * int(output_classes[l][i] == output_classes[l][j])
                )
                suma += a
        all_auc += 1 - (auc / suma)

    return all_auc / unique_target.shape[0]
    # value = 0
    # if len(target) >= 5:
    #     value = roc_auc_score(target, output)
    # return value


def cm(output, target, classes=2):
    """
    cm (short for Confusion Matrix) is a function designed to compute the
    confusion matrix for classification tasks. It creates a matrix that
    summarizes the performance of a classification algorithm by comparing
    predicted labels against true labels. This function is configured to
    handle both binary and multi-class classification tasks.

    :param output (array-like):
        Predicted labels from the model. These are expected to be discrete
        values representing class labels.
    :param target (array-like):
        Ground truth labels corresponding to the predictions.
    :param classes (int, optional):
        The number of classes in the classification task. Default is set
        to 2, which is suitable for binary classification.
    :returns:
        cma (numpy array): The computed confusion matrix, a 2D array where
        each cell [i, j] represents the number of samples of class i that
        were predicted as class j.
    """
    cma = np.zeros((classes, classes))

    # nt = target.numpy()
    for i in range(len(target)):
        # a = np.argmax(output[i])
        a = int(output[i])
        b = int(target[i])
        cma[b, a] += 1

    return cma


def accuracy(weights, pred, target):
    """
    accuracy is a function designed to calculate the weighted accuracy of
    predictions in a classification task. It takes into account the weights
    associated with each prediction, making it particularly suitable for
    datasets where certain samples have more significance than others.

    :param weights (array-like):
        Weights associated with each sample in the dataset, indicating the
        importance or influence of individual samples in the accuracy computation.
    :param pred (array-like):
        Predicted labels from the model. These are expected to be discrete
        values representing class labels.
    :param target (array-like):
        Ground truth labels corresponding to the predictions.
    :returns:
        acc (float): The weighted accuracy of the predictions, a value between
        0 and 1, where 1 indicates perfect accuracy.
    """
    correct_pred = pred == target
    acc = np.sum(correct_pred * weights) / np.sum(weights)
    return acc


def recall(weights, pred, target):
    """
    recall is a function designed to calculate the weighted recall for
    classification tasks. It is capable of handling both binary and multi-class
    scenarios. In multi-class classification, it computes the macro-average
    recall, which averages the recall scores calculated separately for each
    class. The inclusion of weights makes this function particularly suitable
    for datasets where certain samples are more significant than others.

    :param weights (array-like):
        Weights associated with each sample, indicating the importance or
        influence of individual samples in the recall computation.
    :param pred (array-like):
        Predicted labels from the model.
    :param target (array-like):
        Ground truth labels corresponding to the predictions.
    :returns:
        recall (float): The weighted recall of the predictions. In the
        multi-class case, this is the macro-average recall. The value is
        between 0 and 1, where 1 indicates perfect recall.
    """
    if len(np.unique(target)) > 2:
        # Macro averaging
        total_recall = 0
        for i in np.unique(target):
            true_positive = (target == np.ones(target.shape[0]) * i) & (
                pred == np.ones(target.shape[0]) * i
            )
            false_negative = (target == np.ones(target.shape[0]) * i) & (
                pred != np.ones(target.shape[0]) * i
            )
            recall = np.sum(true_positive * weights) / (
                np.sum((true_positive + false_negative) * weights) + np.finfo(float).eps
            )
            total_recall += recall
        return total_recall / np.unique(target).shape[0]
    else:
        true_positive = (target == np.ones(target.shape[0])) & (
            pred == np.ones(target.shape[0])
        )
        false_negative = (target == np.ones(target.shape[0])) & (
            pred == np.zeros(target.shape[0])
        )
        recall = np.sum(true_positive * weights) / (
            np.sum((true_positive + false_negative) * weights) + np.finfo(float).eps
        )
        return recall


def precision(weights, pred, target):
    """
    precision is a function developed to calculate the weighted precision for
    classification tasks, adaptable to both binary and multi-class scenarios.
    In multi-class classification, it employs macro averaging, which calculates
    precision for each class separately and then averages these scores. The
    function's utilization of weights is particularly useful in datasets where
    some samples have more significance than others.

    :param weights (array-like):
        Weights associated with each sample, highlighting the importance or
        influence of individual samples in the precision computation.
    :param pred (array-like):
        Predicted labels from the model.
    :param target (array-like):
        Ground truth labels corresponding to the predictions.
    :returns:
        precision (float): The weighted precision of the predictions. In the
        multi-class case, this is the macro-average precision. The value ranges
        from 0 to 1, where 1 signifies perfect precision.
    """
    if len(np.unique(target)) > 2:
        # Macro averaging
        total_precision = 0
        for i in np.unique(target):
            true_positive = (target == np.ones(target.shape[0]) * i) & (
                pred == np.ones(target.shape[0]) * i
            )
            false_positive = (target != np.ones(target.shape[0]) * i) & (
                pred == np.ones(target.shape[0]) * i
            )
            precision = np.sum(true_positive * weights) / (
                np.sum((true_positive + false_positive) * weights) + np.finfo(float).eps
            )
            total_precision += precision
        return total_precision / np.unique(target).shape[0]
    else:
        true_positive = (target == np.ones(target.shape[0])) & (
            pred == np.ones(target.shape[0])
        )
        false_positive = (target == np.zeros(target.shape[0])) & (
            pred == np.ones(target.shape[0])
        )
        precision = np.sum(true_positive * weights) / (
            np.sum((true_positive + false_positive) * weights) + np.finfo(float).eps
        )
        return precision


def f1(weights, pred, target):
    """
    f1 is a function designed to compute the weighted F1 score for classification
    tasks, adaptable for both binary and multi-class scenarios. The F1 score is a
    harmonic mean of precision and recall, offering a balance between these two
    metrics. It is particularly useful when you need to take both false positives
    and false negatives into account. The function calculates the weighted versions
    of precision and recall and then uses these to compute the F1 score.

    :param weights (array-like):
        Weights associated with each sample in the dataset, emphasizing the
        importance or influence of individual samples in the F1 score computation.
    :param pred (array-like):
        Predicted labels from the model.
    :param target (array-like):
        Ground truth labels corresponding to the predictions.
    :returns:
        f1 (float): The weighted F1 score of the predictions. The value ranges from
        0 to 1, where 1 indicates perfect precision and recall balance.
    """
    pr = precision(weights, pred, target)
    rc = recall(weights, pred, target)
    f1 = 2 * pr * rc / (pr + rc + np.finfo(float).eps)
    return f1


def cross_entropy(weights, output, target):
    """
    cross_entropy is a function designed to calculate the weighted cross-entropy
    loss for classification tasks. Cross-entropy is a widely used loss function
    in classification, particularly useful when dealing with probabilities. This
    function is applicable to both binary and multi-class classification scenarios
    and incorporates sample weights, making it suitable for datasets where certain
    samples have more significance than others.

    :param weights (array-like):
        Weights associated with each sample in the dataset, indicating the
        importance or influence of individual samples in the loss computation.
    :param output (array-like):
        Predicted probabilities from the model. These are continuous values
        typically between 0 and 1.
    :param target (array-like):
        Ground truth labels corresponding to the predictions, usually in a binary
        format (0 or 1) for binary classification tasks.
    :returns:
        The weighted cross-entropy loss, a float value representing the average
        weighted log loss across all samples.
    """
    weights = weights / np.sum(weights)
    yp = np.clip(output, 1e-7, 1 - 1e-7)
    term_1 = target * np.log(yp)
    return -np.sum(weights * term_1)


def compute_regression_metrics(weights, pred, labels):
    """
    compute_regression_metrics is a function tailored to compute key metrics
    for evaluating regression models, specifically Mean Absolute Error (MAE),
    Mean Squared Error (MSE), and R-squared (R²). The function applies weights
    to each sample, making it suitable for datasets where certain samples are
    more significant than others.

    :param weights (array-like):
        Weights associated with each sample, highlighting the importance or
        influence of individual samples in the metric computation.
    :param pred (array-like):
        Predicted values from the regression model.
    :param labels (array-like):
        Ground truth values corresponding to the predictions.
    :returns:
        metrics (dict): A dictionary containing the following key-value pairs
        representing different regression metrics:
            - mae: Weighted Mean Absolute Error.
            - mse: Weighted Mean Squared Error.
            - r2: R-squared value, representing the proportion of variance in
            the dependent variable that is predictable from the independent variables.
    """
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
    metrics = {"mae": mae, "mse": mse, "r2": r2}

    return metrics


def compute_mean_metrics(list, classes=2, metric_type=0):
    """
    compute_mean_metrics is a versatile function designed to compute the mean
    of either classification or regression metrics, depending on the specified
    metric type. It serves as a wrapper that calls either classification_mean_metrics
    or regression_mean_metrics based on the nature of the task. This function is
    particularly useful in scenarios where metrics are computed across multiple
    batches or segments of data, and an aggregated average is needed.

    :param list (list of dicts):
        A list containing dictionaries of metrics computed over different
        segments (e.g., batches of data). Each dictionary in the list should
        contain the same set of metric keys.
    :param classes (int, optional):
        The number of classes in the classification task. This parameter is
        relevant only for classification metrics. Default is set to 2, suitable
        for binary classification.
    :param metric_type (int, optional):
        An integer that specifies the type of metrics to compute. A typical
        convention might be 0 for classification metrics and 1 (or any non-zero
        value) for regression metrics. Default value is 0.
    :returns:
        The mean of the metrics computed by either classification_mean_metrics
        or regression_mean_metrics, depending on the metric_type. The return
        format will be a dictionary of the same structure as the input metric
        dictionaries but with averaged values.
    """
    if metric_type == 0:
        return classification_mean_metrics(list, classes)
    else:
        return regression_mean_metrics(list)


def classification_mean_metrics(dict_list, classes=2):
    """
    classification_mean_metrics is a function specifically designed to compute
    the mean of various classification metrics across multiple data segments,
    such as batches or folds. This function is particularly useful for aggregating
    results from different parts of a dataset to get an overall performance metric.
    It handles both standard metrics and confusion matrices.

    :param dict_list (list of dicts):
        A list containing dictionaries of classification metrics. Each dictionary
        represents metrics computed for a segment of data.
    :param classes (int, optional):
        The number of classes in the classification task. This parameter is
        relevant for the aggregation of confusion matrices. Default is set to 2,
        suitable for binary classification.
    :returns:
        mean_dict (dict): A dictionary containing the mean of each metric. For
        standard metrics, this is a simple average, and for confusion matrices,
        it is an element-wise sum converted to a list.
    """
    mean_dict = {}
    for key in dict_list[0].keys():
        if key == "cm":
            x = np.zeros((classes, classes))
            for d in dict_list:
                for i in range(classes):
                    for j in range(classes):
                        x[i, j] = x[i, j] + d[key][i][j]
            mean_dict[key] = x.tolist()
        else:
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


def regression_mean_metrics(dict_list):
    """
    regression_mean_metrics is a function designed to compute the average of
    various regression metrics across multiple data segments, such as batches
    or folds. This function is particularly useful for aggregating results from
    different parts of a dataset to obtain an overall performance metric for
    regression tasks. It calculates the mean for each metric provided in the
    list of dictionaries.

    :param dict_list (list of dicts):
        A list containing dictionaries of regression metrics. Each dictionary
        represents metrics computed for a segment of data.
    :returns:
        mean_dict (dict): A dictionary containing the mean of each metric. The
        mean is calculated as a simple average of the metric values across all
        provided dictionaries.
    """
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
