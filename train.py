import torch
from ppf.data.dataloader import build_loso_dataloaders
from ppf.models.accelEncoder import AccelEncoder, GyroEncoder
from ppf.models.projector import Projector
from ppf.models.task_head import TaskHead
from ppf.models.identity_probe import IdentityProbe
from ppf.training.multimodal_trainer import MultimodalTrainer


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print("Loading data...")
    train_loader, val_loader = build_loso_dataloaders(
        root_dir="UTD-MHAD", val_subject=8, batch_size=32
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    encoders = {
        "accel": AccelEncoder(),
        "gyro":  GyroEncoder(),
    }
    projectors = {
        "accel": Projector(128),
        "gyro":  Projector(128),
    }
    task_head = TaskHead(hidden_size=512, num_classes=27)
    probe     = IdentityProbe(input_dim=512, output_dim=8)

    trainer = MultimodalTrainer(
        encoders=encoders,
        projectors=projectors,
        task_head=task_head,
        probe=probe,
        device=device,
        total_epochs=100,
        warmup_epochs=10,
        lambda_max=1.0,
    )

    for epoch in range(1, trainer.total_epochs + 1):
        losses  = trainer.train_epoch(train_loader, epoch)
        metrics = trainer.evaluate(val_loader, train_loader)
        print(
            f"Epoch {epoch:2d} | "
            f"λ={losses['lambda']:.3f}  "
            f"task_loss={losses['task_loss']:.4f}  "
            f"probe_loss={losses['probe_loss']:.4f} | "
            f"task_acc={metrics['task_accuracy']:.3f}  "
            f"probe_acc={metrics['probe_accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
