import numpy as np
import torch

def split_calibration_test(smx_val, gt_val, pred_val, X_val, n=100, seed=None):
    """
    Split data into calibration and test sets for conformal prediction.
    smx_val: softmax outputs (N, num_classes)
    gt_val: ground truth labels (N,)
    pred_val: predicted labels (N,)
    X_val: input data (N, ...)
    n: number of calibration samples
    seed: random seed for reproducibility
    Returns: (cal_smx, test_smx, cal_labels, test_labels, model_cal_pred, model_test_pred, X_cal, X_test)
    """
    N = smx_val.shape[0]
    idx = np.array([1] * n + [0] * (N - n)) > 0
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(idx)
    cal_smx, test_smx = smx_val[idx, :], smx_val[~idx, :]
    cal_labels, test_labels = gt_val[idx], gt_val[~idx]
    model_cal_pred, model_test_pred = pred_val[idx], pred_val[~idx]
    X_cal, X_test = X_val[idx], X_val[~idx]
    return cal_smx, test_smx, cal_labels, test_labels, model_cal_pred, model_test_pred, X_cal, X_test

def print_icp_shapes(X_cal, X_test, cal_smx, test_smx, cal_labels, test_labels):
    """
    Print the shapes and dtypes of calibration and test splits for conformal prediction.
    """
    print(f'The shape of the calibration samples {X_cal.shape}, and the test samples {X_test.shape} and with dtype {X_cal.dtype}\n')
    print(f'The shape of the calibration softmax {cal_smx.shape}, and the test softmax {test_smx.shape} and with dtype {cal_smx.dtype}\n')
    print(f'The shape of the calibration labels {cal_labels.shape}, and the test labels {test_labels.shape} and with dtype {cal_labels.dtype}\n') 