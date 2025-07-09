import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, cohen_kappa_score, roc_auc_score, recall_score, precision_score, f1_score
)
import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss):
    """Plot training and validation loss curves."""
    # TODO: Insert plotting logic from notebook
    pass 

def get_predictions(model, dataloader, device):
    """Collect predictions and ground truth from a dataloader."""
    model.eval()
    predictions = []
    gt = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            predicted = torch.max(y_hat.data, 1)[1]
            predictions.append(predicted.cpu())
            gt.append(y.cpu())
    pred_all = torch.cat(predictions, dim=0).numpy()
    gt_all = torch.cat(gt, dim=0).numpy()
    return pred_all, gt_all

def plot_confusion_matrix(pred, gt, class_labels=('0', '1'), cmap='inferno'):
    """Plot a normalized confusion matrix."""
    cm = confusion_matrix(pred, gt, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(class_labels))
    disp.plot(cmap=cmap)
    plt.show()

def compute_metrics(gt, pred):
    """Compute and print classification metrics."""
    acc = accuracy_score(gt, pred)
    bal_acc = balanced_accuracy_score(gt, pred)
    f1 = f1_score(gt, pred, average='weighted')
    report = classification_report(gt, pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:\n", report)
    return {'accuracy': acc, 'balanced_accuracy': bal_acc, 'f1': f1, 'report': report} 

# --- MC Dropout Model Evaluation ---
def load_mc_model(model, path):
    """Load MC Dropout model weights from file."""
    model.load_state_dict(torch.load(path))
    return model

def eval_mc_model(model, dataloader, device):
    """Evaluate MC Dropout model: collect predictions, ground truth, and logits (softmax outputs)."""
    model.eval()
    predictions = []
    gt = []
    logits = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            logits.append(y_hat.cpu())
            predicted = torch.max(y_hat.data, 1)[1]
            predictions.append(predicted.cpu())
            gt.append(y.cpu())
    pred_all = torch.cat(predictions, dim=0)
    gt_all = torch.cat(gt, dim=0)
    logits_all = torch.cat(logits, dim=0)
    return pred_all, gt_all, logits_all

def process_mc_outputs(logits):
    """Exponentiate logits to get probabilities."""
    return torch.exp(logits)

def compute_mc_metrics(gt, pred, class_labels=('0', '1')):
    """Compute accuracy and plot confusion matrix for MC Dropout model."""
    gt_np = gt.cpu().numpy() if torch.is_tensor(gt) else gt
    pred_np = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    acc = accuracy_score(gt_np, pred_np)
    print(f"Accuracy: {acc:.4f}")
    plot_confusion_matrix(pred_np, gt_np, class_labels=class_labels)
    return acc 

def mc_dropout_sampling(model, dataloader, device, num_samples=1000):
    """
    Perform Monte Carlo sampling for uncertainty estimation with an MC Dropout model.
    For each sample, sets dropout layers to train mode, freezes all parameters, and collects logits for each batch.
    Returns a tensor of shape (num_samples, N, num_classes) with all sampled logits.
    """
    import torch.nn as nn
    model.eval()
    all_samples = []
    for i in range(num_samples):
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze dropout layers
        for module in model.children():
            for child in module.children():
                if isinstance(child, nn.Dropout2d):
                    child.train(True)
        if hasattr(model, 'drop'):
            model.drop.train(True)
        sample_logits = []
        with torch.no_grad():
            for X, _ in dataloader:
                X = X.to(device)
                y_hat = model(X)
                sample_logits.append(y_hat.cpu())
        sample_logits = torch.cat(sample_logits, dim=0)
        all_samples.append(sample_logits.unsqueeze(0))
    all_samples = torch.cat(all_samples, dim=0)  # (num_samples, N, num_classes)
    return all_samples 

def stack_and_reshape_mc_samples(samples_list, num_samples, N, num_classes):
    """
    Stack a list of MC sample tensors and reshape to (num_samples, N, num_classes).
    """
    stacked = torch.cat([x for x in samples_list], dim=0)
    return stacked.view(num_samples, N, num_classes)

def get_mc_probabilities(mc_logits):
    """
    Apply torch.exp to MC logits to get probabilities.
    """
    return torch.exp(mc_logits)

def get_mc_std(mc_samples):
    """
    Compute standard deviation across MC samples (dim=0) for uncertainty estimation.
    Returns a tensor of shape (N, num_classes).
    """
    return mc_samples.std(dim=0) 

def mc_dropout_sampling_freeze_bn(model, dataloader, device, num_samples=1000):
    """
    Monte Carlo Dropout sampling with BatchNorm layers frozen (weights/biases not updated, running stats not updated).
    Returns a list of accuracy values (one per sample).
    """
    import torch.nn as nn
    from sklearn.metrics import accuracy_score

    mc_acc = []
    model.train()  # Dropout active, but we'll freeze BN below

    for i in range(num_samples):
        # Freeze BN layers' weights and biases
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False

        predictions = []
        gt = []
        for X_test, y_test in dataloader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            with torch.no_grad():
                y_hat = model(X_test)
                predicted = torch.max(y_hat.data, 1)[1]
                predictions.append(predicted.cpu())
                gt.append(y_test.cpu())
        gt_test = torch.cat(gt, dim=0).numpy()
        pred_test = torch.cat(predictions, dim=0).numpy()
        test_acc = accuracy_score(gt_test, pred_test)
        mc_acc.append(test_acc)
    return mc_acc 