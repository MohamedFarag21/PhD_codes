import numpy as np
import torch

def compute_calibration_scores(cal_smx, cal_labels):
    """
    Compute calibration scores for conformal prediction.
    cal_smx: calibration softmax outputs (n, num_classes)
    cal_labels: calibration labels (n,)
    Returns: calibration scores (n,)
    """
    n = cal_smx.shape[0]
    # Assumes cal_labels are integer indices
    cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
    return cal_scores

def compute_qhat(cal_scores, n, alpha):
    """
    Compute the quantile threshold qhat for conformal prediction.
    cal_scores: calibration scores (n,)
    n: number of calibration samples
    alpha: miscoverage level
    Returns: qhat (float)
    """
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    return qhat

def construct_prediction_sets(test_smx, qhat):
    """
    Construct prediction sets for the test set.
    test_smx: test softmax outputs (N, num_classes)
    qhat: quantile threshold
    Returns: prediction_sets (N, num_classes) boolean array
    """
    prediction_sets = test_smx >= (1 - qhat)
    return prediction_sets

def compute_empirical_coverage(prediction_sets, test_labels):
    """
    Compute empirical coverage of the prediction sets.
    prediction_sets: (N, num_classes) boolean array
    test_labels: (N,) integer array
    Returns: mean_empirical_coverage (float)
    """
    empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), test_labels]
    mean_empirical_coverage = empirical_coverage.sum() / len(empirical_coverage)
    return mean_empirical_coverage

# Example usage (commented):
# alpha = 0.2
# cal_scores = compute_calibration_scores(cal_smx, cal_labels)
# qhat = compute_qhat(cal_scores, n=len(cal_scores), alpha=alpha)
# prediction_sets = construct_prediction_sets(test_smx, qhat)
# mean_empirical_coverage = compute_empirical_coverage(prediction_sets, test_labels)
# print(f"The empirical coverage is: {mean_empirical_coverage}") 