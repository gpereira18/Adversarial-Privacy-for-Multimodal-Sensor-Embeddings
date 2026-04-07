import torch
import torch.nn as nn


class PrivacyProbe(nn.Module):
    """
    Token-level privacy probe — predicts PII (1) or non-PII (0) per token.

    Two-layer MLP applied independently to each token's hidden state.
    Stronger than a single linear layer so the adversarial game is meaningful:
    if a 2-layer probe can't recover PII positions, the information is truly
    scrubbed from the representations. (Ganin et al. 2016 use multi-layer
    domain classifiers for the same reason.)

    Input:  (batch, seq_len, hidden_size)
    Output: (batch, seq_len) — logits per token
    """

    def __init__(self, hidden_size: int = 768, probe_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, probe_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(probe_hidden, 1),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: (batch, seq_len, hidden_size)
        Returns:
            logits: (batch, seq_len) — one PII/non-PII score per token
        """
        return self.net(token_embeddings).squeeze(-1)
