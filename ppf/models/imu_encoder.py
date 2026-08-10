import torch
import torch.nn as nn


class IMUEncoder(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        return torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)


if __name__ == "__main__":
    x = torch.randn(8, 128, 6)
    print(IMUEncoder()(x).shape)
