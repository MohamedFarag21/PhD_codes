import torch
import torch.nn as nn
from torchvision.models import densenet169, DenseNet169_Weights


def conv_bank(in_channels, out_channels):
    """
    Creates a sequence of convolutional, ReLU, BatchNorm, and MaxPool layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        nn.Sequential: Sequential block of layers.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
        nn.ReLU(),
        nn.LazyBatchNorm2d(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
        nn.ReLU(),
        nn.LazyBatchNorm2d(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
        nn.ReLU(),
        nn.LazyBatchNorm2d(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    )

class LeNet(nn.Module):
    """
    LeNet-style convolutional neural network for image classification.

    Args:
        numChannels (int): Number of input channels.
        classes (int): Number of output classes.
    """
    def __init__(self, numChannels, classes):
        super(LeNet, self).__init__()
        self.bank1 = conv_bank(3, 64)
        self.bank2 = conv_bank(64, 128)
        self.bank3 = conv_bank(128, 256)
        self.fc1 = nn.Linear(in_features=256*32*32, out_features=classes)
        self.drop = nn.Dropout(p=0.25)
        [nn.init.xavier_normal_(i.weight) for i in self.bank1 if isinstance(i, nn.Conv2d)]
        [nn.init.xavier_normal_(i.weight) for i in self.bank2 if isinstance(i, nn.Conv2d)]
        [nn.init.xavier_normal_(i.weight) for i in self.bank3 if isinstance(i, nn.Conv2d)]
        nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x):
        """
        Forward pass of the LeNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (N, classes).
        """
        x = self.bank1(x)
        x = self.bank2(x)
        x = self.bank3(x)
        x = self.drop(torch.flatten(x, 1))
        x = self.fc1(x)
        return x


def get_transfer_model(num_classes):
    """
    Loads a pre-trained DenseNet169 model and freezes its parameters.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        model (nn.Module): Modified DenseNet169 model.
    """
    model = densenet169(weights=DenseNet169_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    # TODO: Replace classifier head as in notebook
    return model 

def conv_bank_MCdrop(in_channels, out_channels):
    """
    Creates a sequence of convolutional, Dropout2d, and ReLU layers for MC Dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        nn.Sequential: Sequential block of layers with dropout.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
        nn.Dropout2d(p=0.1),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
        nn.Dropout2d(p=0.1),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same'),
        nn.Dropout2d(p=0.1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
    )

class LeNet_MC(nn.Module):
    """
    LeNet variant with Monte Carlo Dropout for uncertainty estimation.

    Args:
        numChannels (int): Number of input channels.
        classes (int): Number of output classes.
    """
    def __init__(self, numChannels, classes):
        super(LeNet_MC, self).__init__()
        self.bank1 = conv_bank_MCdrop(3, 64)
        self.bank2 = conv_bank_MCdrop(64, 128)
        self.bank3 = conv_bank_MCdrop(128, 256)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(in_features=256*32*32, out_features=classes)
        self.drop = nn.Dropout(p=0.1)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        [nn.init.xavier_normal_(i.weight) for i in self.bank1 if isinstance(i, nn.Conv2d)]
        [nn.init.xavier_normal_(i.weight) for i in self.bank2 if isinstance(i, nn.Conv2d)]
        [nn.init.xavier_normal_(i.weight) for i in self.bank3 if isinstance(i, nn.Conv2d)]
        nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x):
        """
        Forward pass with dropout active at inference for MC sampling.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (N, classes).
        """
        x = self.bn1(self.bank1(x))
        x = self.bn2(self.bank2(x))
        x = self.bank3(x)
        x = self.drop(torch.flatten(x, 1))
        x = self.fc1(x)
        output = self.logSoftmax(x)
        return output 