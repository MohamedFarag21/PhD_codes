import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def cp_check(smx_val, gt_cp, y_cp, n=100, alpha=0.1, r=1000, seed=None):
    """
    Run conformal prediction r times with random splits, returning empirical coverage for each run.
    smx_val: softmax outputs (N, num_classes)
    gt_cp: ground truth labels (N,)
    y_cp: predicted labels (N,)
    n: number of calibration samples
    alpha: miscoverage level
    r: number of repetitions
    seed: random seed for reproducibility
    Returns: list of empirical coverage values (length r)
    """
    final_empirical_coverage = []
    N = smx_val.shape[0]
    rng = np.random.default_rng(seed)
    for i in range(r):
        idx = np.array([1] * n + [0] * (N - n)) > 0
        rng.shuffle(idx)
        cal_smx, test_smx = smx_val[idx, :], smx_val[~idx, :]
        cal_labels, test_labels = gt_cp[idx], gt_cp[~idx]
        model_cal_pred, model_test_pred = y_cp[idx], y_cp[~idx]
        cal_labels, test_labels = cal_labels.astype(int), test_labels.astype(int)
        cal_scores = 1 - cal_smx[np.arange(n), cal_labels]
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        qhat = np.quantile(cal_scores, q_level, method='higher')
        prediction_sets = test_smx >= (1 - qhat)
        empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), test_labels]
        mean_empirical_coverage = empirical_coverage.sum() / len(empirical_coverage)
        final_empirical_coverage.append(mean_empirical_coverage)
    return final_empirical_coverage

def plot_empirical_coverage_hist(final_empirical_coverage, r, alpha, mean_line=None, save_path=None):
    """
    Plot histogram of empirical coverage over r runs, with mean and 1-alpha reference lines.
    final_empirical_coverage: list or array of empirical coverage values
    r: number of runs
    alpha: miscoverage level
    mean_line: value for mean empirical coverage (optional)
    save_path: if provided, saves the figure to this path
    """
    sns.set(style="ticks")
    plt.figure(dpi=600)
    bins = np.linspace(min(final_empirical_coverage), max(final_empirical_coverage), 15)
    sns.histplot(np.array(final_empirical_coverage), kde=True, bins=bins)
    if mean_line is not None:
        plt.axvline(x=mean_line, color='black', linestyle='dashed')
    plt.axvline(x=1 - alpha, color='red', linestyle='solid')
    plt.title(f'Histogram of The Empirical coverage after {r} runs')
    plt.legend(['Coverage distribution', 'Mean empirical average', r'$1-\alpha$'])
    plt.grid()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=600)
    plt.show()

# Example usage (commented):
# final_empirical_coverage = cp_check(smx_val, gt_cp, y_cp, n=100, alpha=0.2, r=1000)
# mean_coverage = np.mean(final_empirical_coverage)
# plot_empirical_coverage_hist(final_empirical_coverage, r=1000, alpha=0.2, mean_line=mean_coverage)
# print(f'The mean empirical coverage after: {r} different data splits is: {mean_coverage}') 