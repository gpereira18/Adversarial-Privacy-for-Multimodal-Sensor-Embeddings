import torch
from torch.utils.data import DataLoader

from ppf.models.encoder import Encoder
from ppf.models.task_head import TaskHead
from ppf.models.privacy_probe import PrivacyProbe
from ppf.models.grl import grad_reverse
from ppf.training.loss import AdversarialLoss
from ppf.training.schedule import lambda_schedule, lr_schedule


class Trainer:
    """
    Adversarial training loop following Ganin et al. (2016).

    Key design choices from the paper:
      - λ warmup: sigmoid schedule from 0 → λ_max (Section 5.2.2)
      - Separate λ: probe trains with fixed λ=1, encoder gets warmup λ
      - LR decay: μ_p = μ_0 / (1 + α*p)^β
      - Single probe step per encoder step (no inner loop)
    """

    def __init__(
        self,
        encoder: Encoder,
        task_head: TaskHead,
        probe: PrivacyProbe,
        loss_fn: AdversarialLoss,
        lr_encoder: float = 2e-5,
        lr_probe: float = 1e-3,
        lambda_max: float = 1.0,
        total_epochs: int = 10,
        warmup_epochs: int = 3,
        device: str = "cpu",
    ):
        self.encoder = encoder.to(device)
        self.task_head = task_head.to(device)
        self.probe = probe.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.lambda_max = lambda_max
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_lambda = 0.0

        # Freeze lower encoder layers to balance adversarial game
        self.encoder.freeze_lower_layers(num_frozen=4)

        # Store initial LRs for scheduling
        self.lr_encoder_init = lr_encoder

        # Only include trainable params in encoder optimizer
        trainable_enc = [p for p in encoder.parameters() if p.requires_grad]
        self.opt_encoder = torch.optim.AdamW(
            trainable_enc + list(task_head.parameters()),
            lr=lr_encoder,
        )
        # Probe LR stays constant (no decay) — probe must stay nimble
        self.opt_probe = torch.optim.Adam(probe.parameters(), lr=lr_probe)

    def _update_schedule(self, epoch: int):
        """Update λ and learning rates based on current epoch.

        During warmup_epochs: λ=0 (probe and task head learn freely).
        After warmup: sigmoid ramp from 0 → λ_max over remaining epochs.
        """
        if epoch <= self.warmup_epochs:
            self.current_lambda = 0.0
        else:
            adv_epoch = epoch - self.warmup_epochs
            adv_total = self.total_epochs - self.warmup_epochs
            self.current_lambda = lambda_schedule(
                adv_epoch, adv_total, self.lambda_max
            )

        # Decay encoder LR (probe LR stays constant — probe must track encoder changes)
        new_lr_enc = lr_schedule(epoch, self.total_epochs, self.lr_encoder_init)
        for pg in self.opt_encoder.param_groups:
            pg["lr"] = new_lr_enc

    @staticmethod
    def _mask_padding(
        tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Zero out padding token embeddings.

        DistilBERT's attention mask prevents real tokens from attending to padding,
        but padding positions still produce non-zero hidden states (they attend to
        real token keys). Explicitly zeroing these out ensures padding never
        influences the GRL or probe — not through loss, not through gradient leakage,
        not through probe capacity waste.

        Shape: tokens (batch, seq_len, hidden) × mask (batch, seq_len, 1) → same
        """
        return tokens * attention_mask.unsqueeze(-1).float()

    def _probe_phase(
        self,
        token_embeddings: torch.Tensor,
        pii_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """Train probe on frozen, padding-masked token embeddings (no GRL).

        Single step per batch — matches Ganin et al. standard alternating SGD.
        """
        self.probe.train()
        emb = token_embeddings.detach()  # already padding-masked by caller

        self.opt_probe.zero_grad()
        probe_logits = self.probe(emb)
        loss = self.loss_fn.probe_loss(probe_logits, pii_labels, attention_mask)
        loss.backward()
        self.opt_probe.step()

        return loss.item()

    def _encoder_phase(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_labels: torch.Tensor,
        pii_labels: torch.Tensor,
    ):
        """Train encoder + task head + probe jointly (Ganin-style).
        GRL reverses probe gradients into encoder, but probe gets normal gradients."""
        self.encoder.train()
        self.task_head.train()
        self.probe.train()

        self.opt_encoder.zero_grad()
        self.opt_probe.zero_grad()
        cls, tokens = self.encoder(input_ids, attention_mask)

        # Zero out padding positions — prevents padding hidden states from
        # leaking into the adversarial game via the GRL or probe
        tokens = self._mask_padding(tokens, attention_mask)

        # Task head gets clean [CLS]
        task_logits = self.task_head(cls)

        # Probe gets GRL-reversed token embeddings (padding already zeroed)
        # GRL reverses gradients for encoder, but probe params get normal gradients
        reversed_tokens = grad_reverse(tokens, self.current_lambda)
        probe_logits = self.probe(reversed_tokens)

        total, task_loss, probe_loss = self.loss_fn.encoder_loss(
            task_logits, task_labels, probe_logits, pii_labels, attention_mask
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        self.opt_encoder.step()
        self.opt_probe.step()

        return task_loss.item(), probe_loss.item()

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
        # Update λ and LR schedules at the start of each epoch
        self._update_schedule(epoch)

        total_task_loss = 0.0
        total_probe_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            task_labels = batch["task_label"].to(self.device)
            pii_labels = batch["pii_token_labels"].to(self.device)

            # Step 1: get token embeddings for probe training (padding-masked)
            with torch.no_grad():
                _, tokens = self.encoder(input_ids, attention_mask)
                tokens = self._mask_padding(tokens, attention_mask)

            # Step 2: train probe on padding-masked embeddings (single step, no GRL)
            self._probe_phase(tokens, pii_labels, attention_mask)

            # Step 3: train encoder with GRL (using warmup λ)
            task_loss, probe_loss = self._encoder_phase(
                input_ids, attention_mask, task_labels, pii_labels
            )

            total_task_loss += task_loss
            total_probe_loss += probe_loss
            n_batches += 1

        return {
            "task_loss": total_task_loss / n_batches,
            "probe_loss": total_probe_loss / n_batches,
            "lambda": self.current_lambda,
        }

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> dict:
        self.encoder.eval()
        self.task_head.eval()
        self.probe.eval()

        task_correct = 0
        probe_tp = 0
        probe_fp = 0
        probe_fn = 0
        probe_tn = 0
        total_samples = 0

        for batch in test_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            task_labels = batch["task_label"].to(self.device)
            pii_labels = batch["pii_token_labels"].to(self.device)

            cls, tokens = self.encoder(input_ids, attention_mask)

            # Mask padding before probe sees embeddings (consistent with training)
            tokens = self._mask_padding(tokens, attention_mask)

            # Task accuracy (sentence-level)
            task_preds = self.task_head(cls).argmax(dim=1)
            task_correct += (task_preds == task_labels).sum().item()
            total_samples += len(task_labels)

            # Probe token-level metrics (only on real tokens)
            probe_preds = (torch.sigmoid(self.probe(tokens)) > 0.5).long()
            mask = attention_mask.bool()
            masked_preds = probe_preds[mask]
            masked_labels = pii_labels[mask]

            probe_tp += ((masked_preds == 1) & (masked_labels == 1)).sum().item()
            probe_fp += ((masked_preds == 1) & (masked_labels == 0)).sum().item()
            probe_fn += ((masked_preds == 0) & (masked_labels == 1)).sum().item()
            probe_tn += ((masked_preds == 0) & (masked_labels == 0)).sum().item()

        total_tokens = probe_tp + probe_fp + probe_fn + probe_tn
        probe_acc = (probe_tp + probe_tn) / total_tokens if total_tokens > 0 else 0
        probe_precision = probe_tp / (probe_tp + probe_fp) if (probe_tp + probe_fp) > 0 else 0
        probe_recall = probe_tp / (probe_tp + probe_fn) if (probe_tp + probe_fn) > 0 else 0
        probe_f1 = (
            2 * probe_precision * probe_recall / (probe_precision + probe_recall)
            if (probe_precision + probe_recall) > 0 else 0
        )

        return {
            "task_accuracy": task_correct / total_samples,
            "probe_accuracy": probe_acc,
            "probe_f1": probe_f1,
            "probe_precision": probe_precision,
            "probe_recall": probe_recall,
        }
