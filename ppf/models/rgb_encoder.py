import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class RGBEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)


if __name__ == "__main__":
    x = torch.randn(8, 16, 3, 112, 112)
    encoder = RGBEncoder()
    print(encoder(x).shape)
