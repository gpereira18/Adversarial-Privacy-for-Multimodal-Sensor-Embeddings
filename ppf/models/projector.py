import torch.nn as nn
import torch

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    x = torch.randn(32, 128)
    projector = Projector(128)
    print(projector(x).shape)