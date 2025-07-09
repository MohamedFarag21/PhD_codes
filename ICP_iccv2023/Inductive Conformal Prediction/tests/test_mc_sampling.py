import torch
from torch.utils.data import DataLoader, TensorDataset
from eval import mc_dropout_sampling

class DummyMCModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
    def forward(self, x):
        return self.fc(x)

def test_mc_dropout_sampling_shape():
    # Dummy data: 10 samples, 2 features
    X = torch.randn(10, 2)
    y = torch.randint(0, 2, (10,))
    loader = DataLoader(TensorDataset(X, y), batch_size=5)
    model = DummyMCModel()
    num_samples = 3
    mc_logits = mc_dropout_sampling(model, loader, device='cpu', num_samples=num_samples)
    assert mc_logits.shape[0] == num_samples, f"Expected {num_samples} MC samples, got {mc_logits.shape[0]}"
    assert mc_logits.shape[1] == 10, f"Expected 10 data points, got {mc_logits.shape[1]}"
    assert mc_logits.shape[2] == 2, f"Expected 2 classes, got {mc_logits.shape[2]}" 