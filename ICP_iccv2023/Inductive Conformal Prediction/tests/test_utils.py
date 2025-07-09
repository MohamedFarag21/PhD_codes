import numpy as np
import torch
from utils import get_dataloaders

def test_get_dataloaders():
    # Create dummy data
    X = np.random.rand(10, 256, 256, 3)
    y = np.random.randint(0, 2, size=(10,))
    tr_dl, vl_dl, ts_dl = get_dataloaders(X, y, X, y, X, y, batch_size=2)
    # Check that DataLoaders are not empty
    assert len(tr_dl.dataset) == 10
    assert len(vl_dl.dataset) == 10
    assert len(ts_dl.dataset) == 10 