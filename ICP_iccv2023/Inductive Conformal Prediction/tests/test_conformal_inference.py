import numpy as np

def test_conformal_inference_single():
    probs = np.array([0.7, 0.2, 0.1])
    qhat = 0.25
    pred_set = np.where(1.0 - probs <= qhat)[0]
    assert set(pred_set) == {0}, f"Expected prediction set {{0}}, got {set(pred_set)}"

def test_conformal_inference_batch():
    probs_batch = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
    qhat = 0.7
    pred_sets = [set(np.where(1.0 - p <= qhat)[0]) for p in probs_batch]
    assert pred_sets[0] == {0, 1, 2}, f"Expected all classes, got {pred_sets[0]}"
    assert pred_sets[1] == {0, 1, 2}, f"Expected all classes, got {pred_sets[1]}" 