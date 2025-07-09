import numpy as np
from conformalization import conformalize

def test_compute_calibration_scores():
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    labels = np.array([0, 1])
    scores = conformalize.compute_calibration_scores(probs, labels)
    assert np.allclose(scores, [0.2, 0.3]), f"Expected [0.2, 0.3], got {scores}"

def test_compute_qhat():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    n = len(scores)
    alpha = 0.25
    qhat = conformalize.compute_qhat(scores, n, alpha)
    assert 0.3 <= qhat <= 0.4, f"qhat should be between 0.3 and 0.4, got {qhat}"

def test_construct_prediction_sets():
    probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    qhat = 0.25
    pred_sets = conformalize.construct_prediction_sets(probs, qhat)
    assert pred_sets.shape == (2, 2)
    assert np.array_equal(pred_sets[0], [True, False])
    assert np.array_equal(pred_sets[1], [False, True])

def test_empirical_coverage():
    pred_sets = np.array([[True, False], [False, True]])
    labels = np.array([0, 1])
    coverage = conformalize.compute_empirical_coverage(pred_sets, labels)
    assert coverage == 1.0, f"Expected coverage 1.0, got {coverage}" 