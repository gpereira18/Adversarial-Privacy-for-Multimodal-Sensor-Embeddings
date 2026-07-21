import torch
import torch.nn.functional as F

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

    train_loader, _ = build_loso_dataloaders("UTD-MHAD", val_subject=8, batch_size=32)
    print(f"Train batches: {len(train_loader)}")

    encoders = {"accel": AccelEncoder(), "gyro": GyroEncoder()}
    projectors = {"accel": Projector(128), "gyro": Projector(128)}
    task_head = TaskHead(hidden_size=512, num_classes=27)
    probe = IdentityProbe(input_dim=512, output_dim=8)

    trainer = MultimodalTrainer(encoders, projectors, task_head, probe, device=device)

    params = (
        list(encoders["accel"].parameters())
        + list(encoders["gyro"].parameters())
        + list(projectors["accel"].parameters())
        + list(projectors["gyro"].parameters())
        + list(task_head.parameters())
        + list(probe.parameters())
    )
    opt = torch.optim.Adam(params, lr=1e-3)

    for epoch in range(1, 31):
        task_correct = probe_correct = total = 0
        for batch in train_loader:
            activity = batch["activity"].to(device)
            subject = batch["subject"].to(device)

            emb = trainer._compute_embeddings(batch)
            task_logits = task_head(emb)
            probe_logits = probe(emb)

            loss = F.cross_entropy(task_logits, activity) + F.cross_entropy(probe_logits, subject)
            opt.zero_grad()
            loss.backward()
            opt.step()

            task_correct += (task_logits.argmax(1) == activity).sum().item()
            probe_correct += (probe_logits.argmax(1) == subject).sum().item()
            total += len(activity)

        print(
            f"Epoch {epoch:2d} | "
            f"task_acc={task_correct / total:.3f}  "
            f"probe_acc={probe_correct / total:.3f}"
        )


if __name__ == "__main__":
    main()
