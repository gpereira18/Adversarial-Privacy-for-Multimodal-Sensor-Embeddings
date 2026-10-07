import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    def __init__(self, lambda_privacy: float = 5.0, pii_pos_weight: float = 1.0):
        super().__init__()
        self.lambda_privacy = lambda_privacy
        self.task_criterion = nn.CrossEntropyLoss()
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
        per_token_loss = self.probe_criterion(probe_logits, pii_labels.float())
        masked_loss = per_token_loss * attention_mask.float()
        return masked_loss.sum() / attention_mask.float().sum().clamp(min=1)

    def encoder_loss(
        self,
        task_logits: torch.Tensor,
        task_labels: torch.Tensor,
        probe_logits: torch.Tensor,
        pii_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return self._masked_probe_loss(probe_logits, pii_labels, attention_mask)
