import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """
    Combined loss for adversarial privacy training with token-level probing.

    Task loss:  CrossEntropy on [CLS] → sentence classification
    Probe loss: BCE on per-token PII predictions, masked to ignore padding

    With GRL, both losses are minimised — the GRL reverses probe gradients
    so the encoder learns to strip PII from token representations.
    """

    def __init__(self, lambda_privacy: float = 5.0, pii_pos_weight: float = 1.0):
        super().__init__()
        self.lambda_privacy = lambda_privacy
        self.task_criterion = nn.CrossEntropyLoss()
        # pos_weight compensates for PII token sparsity (e.g. only 2.5% are PII)
        # reduction='none' so we can mask padding tokens before averaging
        self.register_buffer("_pos_weight", torch.tensor([pii_pos_weight]))
        self.probe_criterion = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=self._pos_weight
        )

    def _masked_probe_loss(
        self,
        probe_logits: torch.Tensor,
        pii_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Token-level BCE masked by attention_mask.
        Only real tokens contribute to the loss — padding is ignored.
        """
        per_token_loss = self.probe_criterion(probe_logits, pii_labels.float())
        # Zero out loss on padding positions
        masked_loss = per_token_loss * attention_mask.float()
        # Average over real tokens only
        return masked_loss.sum() / attention_mask.float().sum().clamp(min=1)

    def encoder_loss(
        self,
        task_logits: torch.Tensor,
        task_labels: torch.Tensor,
        probe_logits: torch.Tensor,
        pii_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encoder step loss. GRL already handles λ scaling and gradient reversal,
        so we just SUM the losses — no λ multiplier here to avoid double-scaling.
        """
        task_loss = self.task_criterion(task_logits, task_labels)
        probe_loss = self._masked_probe_loss(probe_logits, pii_labels, attention_mask)
        total = task_loss + probe_loss
        return total, task_loss, probe_loss

    def probe_loss(
        self,
        probe_logits: torch.Tensor,
        pii_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Probe step loss (no GRL — probe trains normally)."""
        return self._masked_probe_loss(probe_logits, pii_labels, attention_mask)
