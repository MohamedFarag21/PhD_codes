"""
Example experiment script for GrowliFlower Image Classification
Demonstrates data loading, model instantiation, training, and evaluation using the modular codebase.
"""
from utils import set_seed, load_labels, get_file_lists, load_images, get_dataloaders, SEED, DEVICE
from models import LeNet
from train import train_model, EarlyStopper
from eval import compute_metrics, get_predictions
import torch.nn as nn
import torch

# 1. Set random seed
set_seed(SEED)

# 2. Dummy data loading (replace with real data loading in practice)
X = torch.rand(100, 3, 256, 256)
y = torch.randint(0, 2, (100,))
tr_dl, vl_dl, ts_dl = get_dataloaders(X, y, X, y, X, y, batch_size=8)

# 3. Model instantiation
model = LeNet(numChannels=3, classes=2).to(DEVICE)

# 4. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Early stopping
early_stopper = EarlyStopper(patience=5)

# 6. Training
train_model(model, tr_dl, vl_dl, criterion, optimizer, DEVICE, epochs=2, early_stopper=early_stopper)

# 7. Evaluation
pred, gt = get_predictions(model, ts_dl, DEVICE)
compute_metrics(gt, pred) 