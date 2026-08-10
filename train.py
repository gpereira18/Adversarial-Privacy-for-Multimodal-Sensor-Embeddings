import argparse
from pathlib import Path
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
    p.add_argument("--out", type=str, default="checkpoints/model.pt")
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
        "accel": Projector(256),
        "gyro":  Projector(256),
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
        task_acc = metrics["task"]
        probe_acc = metrics["probe"]
        flag = "  <-- COLLAPSED" if metrics["probe_n_classes"] <= 2 else ""
        print(
            f"Epoch {epoch:2d} | λ={losses['lambda']:.3f} | "
            f"task all={task_acc['all']:.3f} "
            f"(accel={task_acc.get('accel', 0):.3f} gyro={task_acc.get('gyro', 0):.3f} rgb={task_acc.get('rgb', 0):.3f}) | "
            f"probe all={probe_acc['all']:.3f} "
            f"(accel={probe_acc.get('accel', 0):.3f} gyro={probe_acc.get('gyro', 0):.3f} rgb={probe_acc.get('rgb', 0):.3f}) | "
            f"pred_classes={metrics['probe_n_classes']} top={metrics['probe_top_share']:.2f}{flag}"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": vars(args),
        "accel": encoders["accel"].state_dict(),
        "gyro": encoders["gyro"].state_dict(),
        "projectors": {k: v.state_dict() for k, v in projectors.items()},
        "task_head": task_head.state_dict(),
        "probe": probe.state_dict(),
        "final_metrics": metrics,
    }, args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
