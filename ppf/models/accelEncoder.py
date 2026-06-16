import torch.nn as nn
import torch

class AccelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(3, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        #x: (batch, 100, 3)
        x = x.transpose(1, 2) # -> (batch, 3, 100)
        x = self.layers(x)
        return x.squeeze(-1)


class GyroEncoder(AccelEncoder):
    pass


if __name__ == "__main__":
    x = torch.randn(32, 100, 3)
    accel_encoder = AccelEncoder()
    print(accel_encoder(x).shape)