import argparse
import torch
from ppf.data.dataloader import build_loso_dataloaders
from ppf.models.accelEncoder import AccelEncoder, GyroEncoder
from ppf.models.rgb_encoder import RGBEncoder
from ppf.models.projector import Projector
from ppf.models.task_head import TaskHead
from ppf.models.identity_probe import IdentityProbe
from ppf.training.multimodal_trainer import MultimodalTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_max", type=float, default=2.13)
    p.add_argument("--lr_probe", type=float, default=0.0045)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Config: lambda_max={args.lambda_max} lr_probe={args.lr_probe} "
          f"k={args.k} warmup_epochs={args.warmup_epochs} epochs={args.epochs}")

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
        "rgb":   RGBEncoder(),
    }
    projectors = {
        "accel": Projector(128),
        "gyro":  Projector(128),
        "rgb":   Projector(512),
    }
    task_head = TaskHead(hidden_size=512, num_classes=27)
    probe     = IdentityProbe(input_dim=512, output_dim=8)

    trainer = MultimodalTrainer(
        encoders=encoders,
        projectors=projectors,
        task_head=task_head,
        probe=probe,
        device=device,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lambda_max=args.lambda_max,
        lr_probe=args.lr_probe,
        k=args.k,
    )

    for epoch in range(1, trainer.total_epochs + 1):
        losses  = trainer.train_epoch(train_loader, epoch)
        metrics = trainer.evaluate(val_loader, train_loader)
        task = metrics["task"]
        probe = metrics["probe"]
        print(
            f"Epoch {epoch:2d} | λ={losses['lambda']:.3f} | "
            f"task all={task['all']:.3f} "
            f"(accel={task.get('accel', 0):.3f} gyro={task.get('gyro', 0):.3f} rgb={task.get('rgb', 0):.3f}) | "
            f"probe all={probe['all']:.3f} "
            f"(accel={probe.get('accel', 0):.3f} gyro={probe.get('gyro', 0):.3f} rgb={probe.get('rgb', 0):.3f})"
        )


if __name__ == "__main__":
    main()
