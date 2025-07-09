from utils import set_seed, load_labels, get_file_lists, load_images, get_dataloaders, SEED, DEVICE, BATCH_SIZE, EPOCHS
from models import LeNet, get_transfer_model
from train import train_model, EarlyStopper
from eval import plot_loss


def main():
    # 1. Set random seed
    set_seed(SEED)

    # 2. Load and preprocess data
    # TODO: Insert data loading and preprocessing logic

    # 3. Create dataloaders
    # TODO: Insert dataloader creation logic

    # 4. Initialize model (choose LeNet or transfer learning)
    # TODO: Insert model initialization logic

    # 5. Set up loss, optimizer, early stopping
    # TODO: Insert setup for criterion, optimizer, early stopper

    # 6. Train model
    # TODO: Insert training loop call

    # 7. Evaluate and plot results
    # TODO: Insert evaluation and plotting logic

if __name__ == '__main__':
    main() 