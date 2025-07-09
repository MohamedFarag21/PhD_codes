import torch
import numpy as np

def prepare_test_softmax_and_labels(test_smx_raw, pred_test, gt_test):
    """
    Prepare test softmax outputs and labels for conformal prediction inference.
    test_smx_raw: list of raw softmax outputs (logits)
    pred_test: predicted labels
    gt_test: ground truth labels
    Returns: test_smx (N, num_classes), y_cp_test (N,), gt_cp_test (N,)
    """
    temp = torch.cat([x for x in test_smx_raw], dim=0)
    temp = torch.exp(temp)
    test_smx = temp
    y_cp_test = torch.Tensor(pred_test).cuda().to(int)
    gt_cp_test = torch.Tensor(gt_test).cuda().to(int)
    return test_smx, y_cp_test, gt_cp_test

def compute_conformal_prediction_sets(test_smx, qhat):
    """
    Compute conformal prediction sets for test data.
    test_smx: (N, num_classes) tensor or array
    qhat: quantile threshold
    Returns: prediction_sets (N, num_classes) boolean array
    """
    prediction_sets = test_smx >= (1 - qhat)
    return prediction_sets

def compute_empirical_coverage(prediction_sets, test_labels):
    """
    Compute empirical coverage of the prediction sets for test data.
    prediction_sets: (N, num_classes) boolean array
    test_labels: (N,) integer array
    Returns: mean_empirical_coverage (float)
    """
    empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), test_labels]
    mean_empirical_coverage = empirical_coverage.sum() / len(empirical_coverage)
    return mean_empirical_coverage

def count_prediction_set_types(pred_set):
    """
    Count the number of single, double, and empty prediction sets in pred_set.
    Returns: single_set, double_set, empty_set
    """
    single_set = 0
    double_set = 0
    empty_set = 0
    for i in pred_set:
        if sum(i) == 1:
            single_set += 1
        elif sum(i) > 1:
            double_set += 1
        else:
            empty_set += 1
    print(f'Number of Single output sets {single_set},\n Double output sets {double_set},\n Empty sets {empty_set}')
    return single_set, double_set, empty_set

def evaluate_single_set_accuracy(y_cp_pred, output_set, y_cp_true, label_strings):
    """
    Evaluate the number of correct single prediction sets for each class.
    Returns: correct_single_noh, correct_single_h, total_correct_single
    """
    correct_single_noh = 0
    correct_single_h = 0
    for i, pred_label in enumerate(y_cp_pred):
        pred_label = pred_label.item() if hasattr(pred_label, 'item') else pred_label
        decision = output_set[i]
        y_cp_true_data = y_cp_true[i].item() if hasattr(y_cp_true[i], 'item') else y_cp_true[i]
        if ((pred_label == 1) and (decision == [label_strings[1]]) and (y_cp_true_data == 1)):
            correct_single_h += 1
        elif ((pred_label == 0) and (decision == [label_strings[0]]) and (y_cp_true_data == 0)):
            correct_single_noh += 1
    total_correct_single = correct_single_h + correct_single_noh
    print(correct_single_noh, correct_single_h, total_correct_single)
    return correct_single_noh, correct_single_h, total_correct_single

def evaluate_double_set_accuracy(y_cp_double, y_cp_double_class_pred, y_cp_double_class_true):
    """
    Evaluate the number of correct double prediction sets.
    Returns: correct_double_set
    """
    correct_double_set = 0
    for i in range(len(y_cp_double)):
        if y_cp_double_class_pred[i] == y_cp_double_class_true[i]:
            correct_double_set += 1
    print(correct_double_set)
    return correct_double_set

def merge_val_test_outputs(smx_val, test_final_smx, gt_cp, gt_cp_test):
    """
    Merge validation and test set softmax outputs and labels for conformal prediction.
    Returns sane_val_test, y_val_test
    """
    sane_val_test = np.concatenate([smx_val.cpu(), test_final_smx.cpu()], axis=0)
    y_val_test = np.concatenate([gt_cp.cpu(), gt_cp_test.cpu()], axis=0)
    return sane_val_test, y_val_test

def cp_check_merged(sane_val_test, y_val_test, n, alpha=0.2, r=1000):
    """
    Run repeated conformal prediction coverage evaluation on merged val/test set.
    Returns list of empirical coverage values.
    """
    final_empirical_coverage = []
    for i in range(r):
        idx = np.array([1] * n + [0] * (sane_val_test.shape[0] - n)) > 0
        np.random.shuffle(idx)
        cal_smx, test_smx = sane_val_test[idx, :], sane_val_test[~idx, :]
        cal_labels, test_labels = y_val_test[idx], y_val_test[~idx]
        cal_smx = torch.Tensor(cal_smx)
        test_smx = torch.Tensor(test_smx)
        cal_labels = torch.Tensor(cal_labels).int()
        test_labels = torch.Tensor(test_labels).int()
        cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = np.quantile(cal_scores, q_level, method='higher')
        prediction_sets = test_smx >= (1 - qhat)
        empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), test_labels]
        mean_empirical_coverage = empirical_coverage.sum() / len(empirical_coverage)
        final_empirical_coverage.append(mean_empirical_coverage)
    return final_empirical_coverage

# Example usage (commented):
# test_smx, y_cp_test, gt_cp_test = prepare_test_softmax_and_labels(test_smx_raw, pred_test, gt_test)
# qhat = ... # compute or load qhat
# prediction_sets = compute_conformal_prediction_sets(test_smx, qhat)
# mean_empirical_coverage = compute_empirical_coverage(prediction_sets, gt_cp_test)
# print(f"The empirical coverage is: {mean_empirical_coverage}")
# single_set, double_set, empty_set = count_prediction_set_types(pred_set)
# correct_single_noh, correct_single_h, total_correct_single = evaluate_single_set_accuracy(y_cp_pred, output_set, y_cp_true, label_strings)
# correct_double_set = evaluate_double_set_accuracy(y_cp_double, y_cp_double_class_pred, y_cp_double_class_true)
# sane_val_test, y_val_test = merge_val_test_outputs(smx_val, test_final_smx, gt_cp, gt_cp_test)
# final_empirical_coverage = cp_check_merged(sane_val_test, y_val_test, n=196, alpha=0.2, r=1000) 