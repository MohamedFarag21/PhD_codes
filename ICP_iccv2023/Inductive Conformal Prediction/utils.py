import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
import random
import glob
import re

# Configuration
SEED = 78
BATCH_SIZE = 32
EPOCHS = 500
IMG_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def load_labels(data_dir, file_no):
    """Load and concatenate label files from the data directory."""
    # TODO: Insert logic from notebook for loading label files
    pass


def get_file_lists(df_all, data_dir):
    """Split data into train, validation, and test file lists."""
    # TODO: Insert logic from notebook for splitting files
    pass


def load_images(file_list, img_size=IMG_SIZE):
    """Load and resize images from file list."""
    # TODO: Insert logic from notebook for loading and resizing images
    pass


def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE):
    """Create DataLoader objects for train, validation, and test sets."""
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    tr_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    vl_dloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    ts_dloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return tr_dloader, vl_dloader, ts_dloader 