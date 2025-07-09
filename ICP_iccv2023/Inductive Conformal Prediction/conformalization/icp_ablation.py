import numpy as np
from sklearn.metrics import accuracy_score

def alpha_ablation(smx_val, gt_cp, test_final_smx, test_final_labels, model_final_test_pred, y_cp, alpha_range=None, n_samples=None):
    """
    Perform ablation over a range of alpha values for conformal prediction.
    For each alpha, compute qhat, prediction sets, and count single, double, and empty sets.
    Compute accuracy for single and double sets, and total accuracy for each alpha.
    Returns arrays of single_pred_sets, double_pred_sets, empty_pred_sets, acc_alpha.
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.1, 1, 100)
    if n_samples is None:
        n_samples = len(y_cp)
    double_pred_sets = np.zeros(len(alpha_range))
    single_pred_sets = np.zeros(len(alpha_range))
    empty_pred_sets = np.zeros(len(alpha_range))
    acc_alpha = np.zeros(len(alpha_range))
    for i, alpha in enumerate(alpha_range):
        double_set_counter = 0
        single_set_counter = 0
        empty_set_counter = 0
        y_cp_true_temp = []
        y_cp_pred_temp = []
        y_cp_double_class_true_temp = []
        y_cp_double_class_pred_temp = []
        n = n_samples
        cal_scores = 1 - smx_val[np.arange(n), gt_cp]
        q_level = np.ceil((n+1)*(1-alpha))/n
        qhat = np.quantile(cal_scores, q_level, method='higher')
        prediction_sets = test_final_smx >= (1-qhat)
        for j, pred_set in enumerate(prediction_sets):
            if sum(pred_set) == 2:
                double_set_counter += 1
                double_pred_sets[i] = double_set_counter
                y_cp_double_class_true_temp.append(test_final_labels[j])
                y_cp_double_class_pred_temp.append(model_final_test_pred[j])
            elif sum(pred_set) == 1:
                single_set_counter += 1
                single_pred_sets[i] = single_set_counter
                y_cp_true_temp.append(test_final_labels[j])
                y_cp_pred_temp.append(model_final_test_pred[j])
            else:
                empty_set_counter += 1
                empty_pred_sets[i] = empty_set_counter
        sing_corr_temp = accuracy_score(y_cp_true_temp, y_cp_pred_temp) if single_set_counter > 0 else 0
        double_corr_temp = accuracy_score(y_cp_double_class_true_temp, y_cp_double_class_pred_temp) if double_set_counter > 0 else 0
        total_acc = (single_set_counter/n) * sing_corr_temp + (double_set_counter/n) * double_corr_temp
        acc_alpha[i] = total_acc
    return single_pred_sets, double_pred_sets, empty_pred_sets, acc_alpha

# Example usage (commented):
# single_pred_sets, double_pred_sets, empty_pred_sets, acc_alpha = alpha_ablation(smx_val, gt_cp, test_final_smx, test_final_labels, model_final_test_pred, y_cp) 