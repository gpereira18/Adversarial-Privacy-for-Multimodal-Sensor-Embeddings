import torch.nn as nn
import torch

class IdentityProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3, output_dim=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    x = torch.randn(32, 512)
    probe = IdentityProbe(512)
    print(probe(x).shape)
