import torch
import torch.nn as nn


class FreshProbe(nn.Module):
    def __init__(self, input_dim=512, hidden=(512, 256), dropout=0.3, output_dim=8):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    x = torch.randn(32, 512)
    print(FreshProbe()(x).shape)
