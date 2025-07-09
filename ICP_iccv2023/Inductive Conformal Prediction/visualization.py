import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

def plot_loss_curves(train_losses, val_losses, title='My Model'):
    """Plot training and validation loss curves."""
    train_hist = torch.tensor(train_losses).detach().cpu().numpy()
    val_hist = torch.tensor(val_losses).detach().cpu().numpy()
    plt.plot(np.array(train_hist))
    plt.plot(np.array(val_hist))
    plt.xlabel("No. of epochs")
    plt.ylabel('Loss function')
    plt.legend(['Train_loss', 'Val_loss'])
    plt.title(title)
    plt.grid()
    plt.show()

def plot_mc_accuracy_distribution(mc_acc, save_path=None):
    """
    Plot the distribution of test accuracy over multiple Monte Carlo samplings.
    mc_acc: list or array of accuracy values from MC runs.
    save_path: if provided, saves the figure to this path.
    """
    sns.set(style="ticks")
    plt.figure(dpi=600)
    bins = np.linspace(min(mc_acc), max(mc_acc), 25)
    sns.histplot(np.array(mc_acc), kde=True, bins=bins)
    plt.axvline(x=np.array(mc_acc).mean(), color='black', linestyle='dashed')
    plt.axvline(x=np.array(mc_acc).mean() + 2 * np.array(mc_acc).std(), color='red', linestyle='dashed')
    plt.axvline(x=np.array(mc_acc).mean() - 2 * np.array(mc_acc).std(), color='red', linestyle='dashed')
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.legend(['Accuracy distribution', r'Mean($\mu$)', r'$\mu\pm2*\sigma$'])
    plt.grid()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=600)
    plt.show()

def plot_accuracy_vs_threshold(prob_class1, gt_labels, pred_labels=None, thresholds=None, save_path=None):
    """
    Plot model accuracy as a function of the decision threshold.
    prob_class1: array of predicted probabilities for class 1.
    gt_labels: ground truth labels (0/1).
    pred_labels: predicted labels at default threshold (for reference line), optional.
    thresholds: array of thresholds to evaluate, default np.linspace(0, 1, 100).
    save_path: if provided, saves the figure to this path.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    acc_values = np.empty(len(thresholds))
    for i, beta in enumerate(thresholds):
        acc_th_pred = [1 if j > beta else 0 for j in prob_class1]
        acc_th = accuracy_score(gt_labels, acc_th_pred)
        acc_values[i] = acc_th
    sns.set(style="ticks")
    plt.figure(dpi=600)
    plt.plot(thresholds, acc_values)
    if pred_labels is not None:
        plt.axhline(y=accuracy_score(gt_labels, pred_labels), color='red', linestyle='dashed')
    plt.xlabel(r'Threshold $\gamma$')
    plt.ylabel('Testing Accuracy')
    plt.legend(['Accuracy', 'Default Threshold Accuracy'])
    plt.title(r'Model Accuracy at Different $\gamma$s')
    plt.grid()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=600)
    plt.show() 