import torch.nn as nn
import torch

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim=512, hidden=None):
        super().__init__()
        if hidden is None:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, output_dim),
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    x = torch.randn(32, 128)
    print(Projector(128)(x).shape)
    print(Projector(128, hidden=256)(x).shape)
