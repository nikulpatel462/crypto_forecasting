import numpy as np


def weighted_correlation(a, b, weights):
    """Evaluation metric copied from the discussion page
    https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/291845

    Excpects columns of actual targets, predictions and asset weights

    Args:
    - a, b: the actual and predicted weights
    - weights: the associated asset weights
    """
    w = np.ravel(weights)
    a = np.ravel(a)
    b = np.ravel(b)

    sum_w = np.sum(w)
    mean_a = np.sum(a * w) / sum_w
    mean_b = np.sum(b * w) / sum_w
    var_a = np.sum(w * np.square(a - mean_a)) / sum_w
    var_b = np.sum(w * np.square(b - mean_b)) / sum_w

    cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
    corr = cov / np.sqrt(var_a * var_b)

    return corr
