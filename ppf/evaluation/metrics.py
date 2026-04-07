import torch
from torch.utils.data import DataLoader

from ppf.models.encoder import Encoder
from ppf.models.task_head import TaskHead
from ppf.models.privacy_probe import PrivacyProbe


@torch.no_grad()
def compute_metrics(
    encoder: Encoder,
    task_head: TaskHead,
    probe: PrivacyProbe,
    data_loader: DataLoader,
    device: str = "cpu",
) -> dict:
    """
    Compute evaluation metrics on a dataset.

    Returns:
        task_accuracy:  fraction of correct task predictions (higher = better utility)
        probe_accuracy: fraction of correct PII predictions by the probe (lower = more private)
        privacy_score:  1 - probe_accuracy (higher = more private, for easy plotting)
    """
    encoder.eval()
    task_head.eval()
    probe.eval()

    task_correct = 0
    probe_correct = 0
    total = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        task_labels = batch["task_label"].to(device)
        pii_labels = batch["pii_label"].to(device)

        embeddings = encoder(input_ids, attention_mask)

        task_preds = task_head(embeddings).argmax(dim=1)
        task_correct += (task_preds == task_labels).sum().item()

        probe_preds = (torch.sigmoid(probe(embeddings)) > 0.5).long()
        probe_correct += (probe_preds == pii_labels).sum().item()

        total += len(task_labels)

    task_acc = task_correct / total
    probe_acc = probe_correct / total

    return {
        "task_accuracy": task_acc,
        "probe_accuracy": probe_acc,
        "privacy_score": 1.0 - probe_acc,
    }
