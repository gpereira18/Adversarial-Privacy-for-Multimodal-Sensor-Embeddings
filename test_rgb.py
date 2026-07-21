import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from ppf.data.dataloader import UTDMHADDataset, mixed_collate
from ppf.models.rgb_encoder import RGBEncoder
from ppf.models.projector import Projector
from ppf.models.task_head import TaskHead
from ppf.models.identity_probe import IdentityProbe


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    ds = UTDMHADDataset("UTD-MHAD")
    train_idx = [i for i, s in enumerate(ds.samples) if s["modality"] == "rgb" and s["subject"] != 8]
    val_idx = [i for i, s in enumerate(ds.samples) if s["modality"] == "rgb" and s["subject"] == 8]
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=True, collate_fn=mixed_collate)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=32, shuffle=False, collate_fn=mixed_collate)
    print(f"RGB train samples: {len(train_idx)}, val samples: {len(val_idx)}")

    enc = RGBEncoder().to(device)
    proj = Projector(128).to(device)
    task_head = TaskHead(512, 27).to(device)
    probe = IdentityProbe(512, output_dim=8).to(device)

    params = list(enc.parameters()) + list(proj.parameters()) + list(task_head.parameters()) + list(probe.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)

    mods = [enc, proj, task_head, probe]

    def run(loader, train):
        for m in mods:
            m.train() if train else m.eval()
        task_c = probe_c = total = 0
        for batch in loader:
            x = torch.stack(batch["input_tensor"]).to(device)
            act = batch["activity"].to(device)
            sub = batch["subject"].to(device)
            with torch.set_grad_enabled(train):
                emb = proj(enc(x))
                task_logits = task_head(emb)
                probe_logits = probe(emb)
                loss = F.cross_entropy(task_logits, act) + F.cross_entropy(probe_logits, sub)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            task_c += (task_logits.argmax(1) == act).sum().item()
            probe_c += (probe_logits.argmax(1) == sub).sum().item()
            total += len(act)
        return task_c / total, probe_c / total

    for epoch in range(1, 41):
        tr_task, tr_probe = run(train_loader, True)
        with torch.no_grad():
            va_task, va_probe = run(val_loader, False)
        print(
            f"Epoch {epoch:2d} | "
            f"train_task={tr_task:.3f}  val_task={va_task:.3f} | "
            f"train_probe={tr_probe:.3f}  val_probe={va_probe:.3f}"
        )


if __name__ == "__main__":
    main()
