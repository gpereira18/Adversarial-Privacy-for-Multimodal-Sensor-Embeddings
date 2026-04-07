import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """Linear classifier on [CLS] embedding → task logits."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(cls_embedding)
