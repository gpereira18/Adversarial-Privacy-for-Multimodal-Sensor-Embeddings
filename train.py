"""
Entry point: loads config, builds models, runs adversarial training loop.

Usage:
    python train.py                          # full training with default.yaml
    python train.py --max_train_samples 500  # quick test with 500 samples
"""

import argparse
import yaml
import torch
import os
from datetime import datetime

from ppf.data.dataset import build_dataloaders
from ppf.models.encoder import Encoder
from ppf.models.task_head import TaskHead
from ppf.models.privacy_probe import PrivacyProbe
from ppf.training.loss import AdversarialLoss
from ppf.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    max_train = args.max_train_samples or cfg["data"].get("max_train_samples")
    max_test = args.max_test_samples or cfg["data"].get("max_test_samples")

    # Data
    print("Loading data...")
    train_loader, test_loader, pii_pos_weight = build_dataloaders(
        dataset_name=cfg["data"]["dataset"],
        tokenizer_name=cfg["model"]["encoder"],
        max_length=cfg["data"]["max_length"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_train_samples=max_train,
        max_test_samples=max_test,
    )

    # Models
    encoder = Encoder(model_name=cfg["model"]["encoder"])
    hidden_size = encoder.hidden_size

    task_head = TaskHead(
        hidden_size=hidden_size,
        num_classes=cfg["model"]["num_classes"],
    )
    probe = PrivacyProbe(hidden_size=hidden_size)

    loss_fn = AdversarialLoss(
        lambda_privacy=cfg["training"]["lambda_max"],
        pii_pos_weight=pii_pos_weight,
    )

    epochs = cfg["training"]["epochs"]

    trainer = Trainer(
        encoder=encoder,
        task_head=task_head,
        probe=probe,
        loss_fn=loss_fn,
        lr_encoder=cfg["training"]["lr_encoder"],
        lr_probe=cfg["training"]["lr_probe"],
        lambda_max=cfg["training"]["lambda_max"],
        total_epochs=epochs,
        warmup_epochs=cfg["training"]["warmup_epochs"],
        device=device,
    )

    # Set up log file
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/train_{timestamp}.log"
    log_file = open(log_path, "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Device: {device}")
    log(f"Train samples: {max_train}, Test samples: {max_test}")
    log(f"Config: epochs={epochs}, lambda_max={cfg['training']['lambda_max']}, "
        f"warmup={cfg['training']['warmup_epochs']}, "
        f"lr_enc={cfg['training']['lr_encoder']}, lr_probe={cfg['training']['lr_probe']}")
    log(f"Log file: {log_path}")
    log("")

    # Training loop
    for epoch in range(1, epochs + 1):
        losses = trainer.train_epoch(train_loader, epoch)
        metrics = trainer.evaluate(test_loader)

        line = (
            f"Epoch {epoch:2d} | "
            f"λ={losses['lambda']:.3f}  "
            f"task_loss={losses['task_loss']:.4f}  "
            f"probe_loss={losses['probe_loss']:.4f} | "
            f"task_acc={metrics['task_accuracy']:.3f}  "
            f"probe_f1={metrics['probe_f1']:.3f}  "
            f"probe_acc={metrics['probe_accuracy']:.3f}"
        )
        log(line)

    log_file.close()
    print(f"\nResults saved to {log_path}")


if __name__ == "__main__":
    main()
