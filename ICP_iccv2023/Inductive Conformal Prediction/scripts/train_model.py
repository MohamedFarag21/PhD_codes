import argparse
import torch
from data_preparation import tr_dloader, vl_dloader
from models import LeNet
from train import train_model
import torch.nn as nn
import torch.optim as optim


def main(args):
    """
    Command-line interface for training a model.
    Args:
        args: Parsed command-line arguments.
    """
    # Select device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Instantiate model
    model = LeNet(numChannels=3, classes=args.num_classes)
    model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    print(f"Training {args.model} for {args.epochs} epochs on {device}...")
    train_losses, val_losses, trn_accuracy, val_accuracy = train_model(
        model, tr_dloader, vl_dloader, criterion, optimizer, device, args.epochs
    )

    # Save the trained model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    # Print final metrics
    print(f"Final Training Accuracy: {trn_accuracy[-1]:.2f}%")
    print(f"Final Validation Accuracy: {val_accuracy[-1]:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model from the command line.")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (set in data_preparation)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    parser.add_argument('--model', type=str, default='LeNet', help='Model type (default: LeNet)')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='trained_model.pt', help='Path to save the trained model')
    args = parser.parse_args()
    main(args) 