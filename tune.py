import optuna
import torch
from ppf.data.dataloader import build_loso_dataloaders
from ppf.models.imu_encoder import IMUEncoder
from ppf.models.rgb_encoder import RGBEncoder
from ppf.models.projector import Projector
from ppf.models.task_head import TaskHead
from ppf.models.identity_probe import IdentityProbe
from ppf.training.multimodal_trainer import MultimodalTrainer

TRIAL_EPOCHS = 60
COLLAPSE_TOP_SHARE = 0.5
N_TRIALS = 30

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


def objective(trial):
    lambda_max = trial.suggest_float("lambda_max", 0.5, 5.0)
    lr_probe = trial.suggest_float("lr_probe", 1e-4, 5e-3, log=True)
    lr_encoder = trial.suggest_float("lr_encoder", 1e-4, 3e-3, log=True)
    k = trial.suggest_int("k", 1, 10)
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 30)

    encoders = {
        "imu": IMUEncoder(),
        "rgb": RGBEncoder(),
    }
    projectors = {
        "imu": Projector(256),
        "rgb": Projector(512, hidden=512),
    }
    task_head = TaskHead(hidden_size=512, num_classes=27)
    probe     = IdentityProbe(input_dim=512, output_dim=8)

    trainer = MultimodalTrainer(
        encoders=encoders,
        projectors=projectors,
        task_head=task_head,
        probe=probe,
        device=device,
        total_epochs=TRIAL_EPOCHS,
        warmup_epochs=warmup_epochs,
        lambda_max=lambda_max,
        lr_probe=lr_probe,
        lr_encoder=lr_encoder,
        k=k,
    )

    task_history = []
    probe_history = []
    top_history = []
    for epoch in range(1, trainer.total_epochs + 1):
        trainer.train_epoch(train_loader, epoch)
        metrics = trainer.evaluate(val_loader, train_loader)
        task_history.append(metrics["task"]["all"])
        probe_history.append(metrics["probe"]["all"])
        top_history.append(metrics["probe_top_share"])

    top_score = sum(top_history[-5:]) / 5
    task_score = sum(task_history[-5:]) / 5
    probe_score = sum(probe_history[-5:]) / 5

    trial.set_user_attr("top_share", top_score)
    if top_score > COLLAPSE_TOP_SHARE:
        print(f"Trial {trial.number}: probe collapsed (top={top_score:.2f}), rejected")
        return 0.0, 1.0

    return task_score, probe_score


if __name__ == "__main__":
    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n=== Pareto front ===")
    for t in study.best_trials:
        task_score, probe_score = t.values
        top = t.user_attrs.get("top_share", float("nan"))
        print(f"Trial {t.number}: task={task_score:.4f} probe={probe_score:.4f} "
              f"top={top:.2f} params={t.params}")
