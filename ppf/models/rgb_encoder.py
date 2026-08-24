import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class RGBEncoder(nn.Module):
    def __init__(self, unfreeze="none"):
        super().__init__()
        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.unfreeze = unfreeze

        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze == "layer4":
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True
        elif unfreeze == "all":
            for p in self.backbone.parameters():
                p.requires_grad = True

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)


if __name__ == "__main__":
    x = torch.randn(2, 32, 3, 112, 112)
    for mode in ("none", "layer4", "all"):
        enc = RGBEncoder(unfreeze=mode)
        n = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        print(f"{mode:7s} -> out {tuple(enc(x).shape)}  trainable={n/1e6:.1f}M")
