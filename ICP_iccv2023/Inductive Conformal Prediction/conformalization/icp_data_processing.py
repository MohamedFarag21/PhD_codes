"""
icp_data_processing.py
---------------------
Data processing utilities for Inductive Conformal Prediction (ICP) experiments.
Handles splitting data into proper training, calibration, and test sets for conformal prediction.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_for_icp(X, y, calib_size=0.2, test_size=0.2, random_state=42):
    """
    Split data into proper training, calibration, and test sets for ICP.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Label array.
        calib_size (float): Fraction for calibration set.
        test_size (float): Fraction for test set.
        random_state (int): Random seed.

    Returns:
        tuple: (X_train, X_calib, X_test, y_train, y_calib, y_test)
    """
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    # Then split the remaining into train/calib
    calib_frac = calib_size / (1 - test_size)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=calib_frac, random_state=random_state, stratify=y_temp)
    return X_train, X_calib, X_test, y_train, y_calib, y_test 