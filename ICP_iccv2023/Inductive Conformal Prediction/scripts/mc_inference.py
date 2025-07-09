import argparse
import torch
import numpy as np
from data_preparation import ts_dloader
from models import LeNet_MC
from eval import mc_dropout_sampling


def main(args):
    """
    Command-line interface for Monte Carlo Dropout inference.
    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load MC Dropout model
    model = LeNet_MC(numChannels=3, classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.to(device)
    model.eval()

    # Select dataloader (default: test set)
    dataloader = ts_dloader  # You can modify to support other splits if needed

    # Run MC Dropout sampling
    print(f"Running MC Dropout inference with {args.num_samples} samples on {args.device}...")
    mc_logits = mc_dropout_sampling(model, dataloader, device, num_samples=args.num_samples)
    mc_probs = torch.exp(mc_logits).numpy()  # Convert logits to probabilities

    # Save outputs
    np.savez(args.output, mc_logits=mc_logits.numpy(), mc_probs=mc_probs)
    print(f"MC Dropout outputs saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Dropout inference script.")
    parser.add_argument('--model_weights', type=str, required=True, help='Path to trained MC Dropout model weights (.pt)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of MC samples')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    parser.add_argument('--output', type=str, default='mc_outputs.npz', help='Path to save MC outputs (npz)')
    args = parser.parse_args()
    main(args) 