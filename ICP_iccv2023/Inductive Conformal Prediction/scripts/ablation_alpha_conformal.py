import argparse
import torch
import numpy as np
from data_preparation import vl_dloader, ts_dloader
from models import LeNet
from conformalization.icp_ablation import alpha_ablation


def main(args):
    """
    Command-line interface for ablation study over alpha for conformal prediction.
    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = LeNet(numChannels=3, classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.to(device)
    model.eval()

    # Get calibration and test set softmax outputs and labels
    cal_logits = []
    cal_labels = []
    with torch.no_grad():
        for X, y in vl_dloader:
            X = X.to(device)
            logits = model(X)
            cal_logits.append(torch.softmax(logits, dim=1).cpu().numpy())
            cal_labels.append(y.cpu().numpy())
    cal_probs = np.concatenate(cal_logits, axis=0)
    cal_labels = np.concatenate(cal_labels, axis=0)

    test_logits = []
    test_labels = []
    model_test_pred = []
    with torch.no_grad():
        for X, y in ts_dloader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            test_logits.append(probs.cpu().numpy())
            test_labels.append(y.cpu().numpy())
            model_test_pred.append(torch.argmax(probs, dim=1).cpu().numpy())
    test_probs = np.concatenate(test_logits, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    model_test_pred = np.concatenate(model_test_pred, axis=0)

    # Alpha range
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)

    # Run ablation
    single_pred_sets, double_pred_sets, empty_pred_sets, acc_alpha = alpha_ablation(
        cal_probs, cal_labels, test_probs, test_labels, model_test_pred, cal_labels, alpha_range=alpha_range, n_samples=len(cal_labels)
    )

    # Print or save results
    print("Ablation study over alpha:")
    for i, alpha in enumerate(alpha_range):
        print(f"alpha={alpha:.3f} | single: {single_pred_sets[i]:.0f}, double: {double_pred_sets[i]:.0f}, empty: {empty_pred_sets[i]:.0f}, acc: {acc_alpha[i]:.3f}")

    if args.output:
        np.savez(args.output, alpha_range=alpha_range, single_pred_sets=single_pred_sets, double_pred_sets=double_pred_sets, empty_pred_sets=empty_pred_sets, acc_alpha=acc_alpha)
        print(f"Ablation results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study over alpha for conformal prediction.")
    parser.add_argument('--model_weights', type=str, required=True, help='Path to trained model weights (.pt)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--alpha_points', type=int, default=20, help='Number of points in alpha range')
    parser.add_argument('--alpha_min', type=float, default=0.01, help='Minimum alpha value')
    parser.add_argument('--alpha_max', type=float, default=0.5, help='Maximum alpha value')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    parser.add_argument('--output', type=str, default='', help='Path to save ablation results (npz)')
    args = parser.parse_args()
    main(args) 