import argparse
import torch
import numpy as np
from data_preparation import vl_dloader, ts_dloader
from models import LeNet
from eval import get_predictions
from conformalization import conformalize


def main(args):
    """
    Command-line interface for applying conformal prediction to a trained model.
    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = LeNet(numChannels=3, classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.to(device)
    model.eval()

    # Get predictions and softmax for calibration and test sets
    # Calibration set
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

    # Test set
    test_logits = []
    test_labels = []
    with torch.no_grad():
        for X, y in ts_dloader:
            X = X.to(device)
            logits = model(X)
            test_logits.append(torch.softmax(logits, dim=1).cpu().numpy())
            test_labels.append(y.cpu().numpy())
    test_probs = np.concatenate(test_logits, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # Conformalization
    cal_scores = conformalize.compute_calibration_scores(cal_probs, cal_labels)
    qhat = conformalize.compute_qhat(cal_scores, len(cal_scores), args.alpha)
    prediction_sets = conformalize.construct_prediction_sets(test_probs, qhat)
    coverage = conformalize.compute_empirical_coverage(prediction_sets, test_labels)
    avg_set_size = np.mean(np.sum(prediction_sets, axis=1))

    print(f"Conformalization results (alpha={args.alpha}):")
    print(f"  Empirical coverage: {coverage:.3f}")
    print(f"  Average prediction set size: {avg_set_size:.3f}")

    # Optionally save results
    if args.output:
        np.savez(args.output, prediction_sets=prediction_sets, test_labels=test_labels, coverage=coverage, avg_set_size=avg_set_size)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply conformal prediction to a trained model.")
    parser.add_argument('--model_weights', type=str, required=True, help='Path to trained model weights (.pt)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level (1 - desired coverage)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    parser.add_argument('--output', type=str, default='', help='Path to save conformalization results (npz)')
    args = parser.parse_args()
    main(args) 