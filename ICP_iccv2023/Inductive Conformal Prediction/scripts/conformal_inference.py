import argparse
import torch
import numpy as np
from PIL import Image
from models import LeNet
from conformalization import conformalize
from data_preparation import ts_dloader


def preprocess_image(image_path, img_height=256, img_width=256):
    """
    Load and preprocess a single image for model inference.
    Args:
        image_path (str): Path to the image file.
        img_height (int): Height to resize image.
        img_width (int): Width to resize image.
    Returns:
        torch.Tensor: Preprocessed image tensor (1, 3, H, W)
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_height, img_width))
    arr = np.array(img) / 255.0
    arr = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return arr


def main(args):
    """
    Command-line interface for conformal prediction inference.
    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = LeNet(numChannels=3, classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.to(device)
    model.eval()

    # Get or compute qhat
    if args.qhat is not None:
        qhat = args.qhat
    elif args.calib_probs is not None and args.calib_labels is not None:
        cal_probs = np.load(args.calib_probs)
        cal_labels = np.load(args.calib_labels)
        cal_scores = conformalize.compute_calibration_scores(cal_probs, cal_labels)
        qhat = conformalize.compute_qhat(cal_scores, len(cal_scores), args.alpha)
    else:
        raise ValueError("You must provide either --qhat or both --calib_probs and --calib_labels (and --alpha)")

    # Single image inference
    if args.image:
        img_tensor = preprocess_image(args.image).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred_set = np.where(1.0 - probs <= qhat)[0]
        print(f"Conformal prediction set for image {args.image}: {pred_set}")
        if args.output:
            np.savez(args.output, pred_set=pred_set, probs=probs, qhat=qhat)
            print(f"Results saved to {args.output}")
    # Batch inference from dataloader
    elif args.batch:
        all_pred_sets = []
        all_probs = []
        with torch.no_grad():
            for X, _ in ts_dloader:
                X = X.to(device)
                logits = model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for p in probs:
                    pred_set = np.where(1.0 - p <= qhat)[0]
                    all_pred_sets.append(pred_set)
                    all_probs.append(p)
        print(f"Conformal prediction sets for batch: {all_pred_sets}")
        if args.output:
            np.savez(args.output, pred_sets=all_pred_sets, probs=all_probs, qhat=qhat)
            print(f"Results saved to {args.output}")
    else:
        print("Please specify either --image for single image or --batch for batch inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformal prediction inference script.")
    parser.add_argument('--model_weights', type=str, required=True, help='Path to trained model weights (.pt)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--qhat', type=float, default=None, help='Conformal threshold qhat (if known)')
    parser.add_argument('--calib_probs', type=str, default=None, help='Path to calibration set probabilities (npy)')
    parser.add_argument('--calib_labels', type=str, default=None, help='Path to calibration set labels (npy)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for qhat computation (if needed)')
    parser.add_argument('--image', type=str, default=None, help='Path to a single image for inference')
    parser.add_argument('--batch', action='store_true', help='Run inference on batch from test dataloader')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    parser.add_argument('--output', type=str, default='', help='Path to save inference results (npz)')
    args = parser.parse_args()
    main(args) 