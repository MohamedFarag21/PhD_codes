import numpy as np
import torch

def compute_calibration_scores(cal_smx, cal_labels):
    """
    Compute calibration scores for conformal prediction.

    Args:
        cal_smx (np.ndarray): Calibration softmax outputs (n, num_classes).
        cal_labels (np.ndarray): Calibration labels (n,).

    Returns:
        np.ndarray: Calibration scores (n,).
    """
    n = cal_smx.shape[0]
    # Assumes cal_labels are integer indices
    cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
    return cal_scores

def compute_qhat(cal_scores, n, alpha):
    """
    Compute the quantile threshold qhat for conformal prediction.

    Args:
        cal_scores (np.ndarray): Calibration scores (n,).
        n (int): Number of calibration samples.
        alpha (float): Miscoverage level.

    Returns:
        float: qhat (quantile threshold).
    """
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    return qhat

def construct_prediction_sets(test_smx, qhat):
    """
    Construct prediction sets for the test set.

    Args:
        test_smx (np.ndarray): Test softmax outputs (N, num_classes).
        qhat (float): Quantile threshold.

    Returns:
        np.ndarray: Boolean array (N, num_classes) indicating prediction sets.
    """
    prediction_sets = test_smx >= (1 - qhat)
    return prediction_sets

def compute_empirical_coverage(prediction_sets, test_labels):
    """
    Compute empirical coverage of the prediction sets.

    Args:
        prediction_sets (np.ndarray): Boolean array (N, num_classes) for prediction sets.
        test_labels (np.ndarray): Integer array (N,) of true labels.

    Returns:
        float: Mean empirical coverage.
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