import torch
import numpy as np

class EarlyStopper:
    """
    Early stopping utility to halt training when validation loss stops improving.

    Args:
        patience (int): Number of epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as improvement.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss, model=None, save_path=None):
        """
        Checks if early stopping condition is met.

        Args:
            validation_loss (float): Current validation loss.
            model (nn.Module, optional): Model to save if improved.
            save_path (str, optional): Path to save the model.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            if model is not None and save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f'Saving the best weights at validation loss: {self.min_validation_loss}\n\n')
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_model(model, tr_dloader, vl_dloader, criterion, optimizer, device, epochs, early_stopper=None, save_path=None):
    """
    Train a model with optional early stopping and save best weights.

    Args:
        model (nn.Module): The neural network to train.
        tr_dloader (DataLoader): DataLoader for training data.
        vl_dloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to train on ('cpu' or 'cuda').
        epochs (int): Maximum number of epochs.
        early_stopper (EarlyStopper, optional): Early stopping utility.
        save_path (str, optional): Path to save best model.

    Returns:
        tuple: (train_losses, val_losses, trn_accuracy, val_accuracy)
    """
    torch.manual_seed(78)
    train_losses = []
    val_losses = []
    trn_accuracy = []
    val_accuracy = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in tr_dloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(tr_dloader)
        train_acc = 100.0 * train_correct / len(tr_dloader.dataset)
        train_losses.append(train_loss)
        trn_accuracy.append(train_acc)

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        with torch.no_grad():
            for images, labels in vl_dloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                valid_correct += (predicted == labels).sum().item()

        valid_loss /= len(vl_dloader)
        valid_acc = 100.0 * valid_correct / len(vl_dloader.dataset)
        val_losses.append(valid_loss)
        val_accuracy.append(valid_acc)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
              f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.2f}%")

        if early_stopper is not None:
            if early_stopper.early_stop(valid_loss, model=model, save_path=save_path):
                break

    return train_losses, val_losses, trn_accuracy, val_accuracy


def train_mc_model(model, tr_dloader, vl_dloader, criterion, optimizer, device, epochs, early_stopper=None, save_path='mymodel_mc_256_bottleneck_balancing_correct.pt'):
    """
    Train a Monte Carlo Dropout model (e.g., LeNet_MC) with the same logic as train_model.
    This function is provided for clarity and future MC-specific extensions.

    Args:
        model (nn.Module): The MC Dropout model to train.
        tr_dloader (DataLoader): DataLoader for training data.
        vl_dloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to train on ('cpu' or 'cuda').
        epochs (int): Maximum number of epochs.
        early_stopper (EarlyStopper, optional): Early stopping utility.
        save_path (str, optional): Path to save best model.

    Returns:
        tuple: (train_losses, val_losses, trn_accuracy, val_accuracy)
    """
    return train_model(model, tr_dloader, vl_dloader, criterion, optimizer, device, epochs, early_stopper, save_path) 