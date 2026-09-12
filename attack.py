import argparse
from collections import Counter

import torch
import torch.nn.functional as F

from ppf.data.dataloader import UTDMHADDataset, compute_normalization_stats
from ppf.models.imu_encoder import IMUEncoder
from ppf.models.rgb_encoder import RGBEncoder
from ppf.models.projector import Projector
from ppf.models.fresh_probe import FreshProbe


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--root", type=str, default="UTD-MHAD")
    p.add_argument("--val_subject", type=int, default=8)
    p.add_argument("--test_trial", type=int, default=4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def trial_of(sample):
    return int(sample["file"].name.split("_")[2][1:])


def build_frozen(ckpt, device):
    rgb_hidden = ckpt["config"].get("rgb_hidden", 512) or None
    encoders = {"imu": IMUEncoder(), "rgb": RGBEncoder()}
    projectors = {"imu": Projector(256), "rgb": Projector(512, hidden=rgb_hidden)}

    encoders["imu"].load_state_dict(ckpt["imu"])
    if "rgb" in ckpt:
        encoders["rgb"].load_state_dict(ckpt["rgb"])
        print("loaded fine-tuned RGB backbone from checkpoint")
    else:
        print("RGB backbone was frozen; using pretrained Kinetics weights")
    for name, proj in projectors.items():
        proj.load_state_dict(ckpt["projectors"][name])

    for m in list(encoders.values()) + list(projectors.values()):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad = False
    return encoders, projectors


@torch.no_grad()
def embed_all(dataset, indices, encoders, projectors, device, batch_size=32):
    by_mod = {}
    for i in indices:
        by_mod.setdefault(dataset.samples[i]["modality"], []).append(i)

    embs, subjects, trials, mods = [], [], [], []
    for modality, idxs in by_mod.items():
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start:start + batch_size]
            x = torch.stack([dataset[i]["input_tensor"] for i in chunk]).to(device)
            e = projectors[modality](encoders[modality](x))
            embs.append(e.cpu())
            for i in chunk:
                s = dataset.samples[i]
                subjects.append(s["subject"] - 1)
                trials.append(trial_of(s))
                mods.append(modality)
    return torch.cat(embs), torch.tensor(subjects), trials, mods


def accuracy_by_modality(preds, labels, mods):
    correct, total = {}, {}
    for p, l, m in zip(preds.tolist(), labels.tolist(), mods):
        total[m] = total.get(m, 0) + 1
        correct[m] = correct.get(m, 0) + int(p == l)
    out = {m: correct[m] / total[m] for m in total}
    out["all"] = sum(correct.values()) / sum(total.values())
    return out


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Trained with: {ckpt['config']}")

    dataset = UTDMHADDataset(args.root)
    train_idx = [i for i, s in enumerate(dataset.samples) if s["subject"] != args.val_subject]
    mean, std = compute_normalization_stats(dataset, train_idx)
    dataset.mean, dataset.std = mean, std

    encoders, projectors = build_frozen(ckpt, device)

    print("Computing embeddings...")
    X, y, trials, mods = embed_all(dataset, train_idx, encoders, projectors, device)

    tr = [i for i, t in enumerate(trials) if t != args.test_trial]
    te = [i for i, t in enumerate(trials) if t == args.test_trial]

    raw_norm = X.norm(dim=1).mean().item()
    mu = X[tr].mean(dim=0, keepdim=True)
    sd = X[tr].std(dim=0, keepdim=True) + 1e-6
    X = (X - mu) / sd
    print(f"standardised attacker inputs: raw mean-norm={raw_norm:.1f} -> "
          f"standardised mean-norm={X.norm(dim=1).mean().item():.1f}")

    X_tr, y_tr = X[tr].to(device), y[tr].to(device)
    X_te, y_te = X[te].to(device), y[te].to(device)
    mods_te = [mods[i] for i in te]

    counts = Counter(y_te.tolist())
    majority = max(counts.values()) / len(te)
    print(f"attack train={len(tr)}  test={len(te)}  "
          f"subjects={len(set(y.tolist()))}  majority-class baseline={majority:.3f}")

    probe = FreshProbe(input_dim=X.shape[1], output_dim=int(y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        probe.train()
        perm = torch.randperm(len(tr), device=device)
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            loss = F.cross_entropy(probe(X_tr[idx]), y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(X_te).argmax(dim=1)
            acc = accuracy_by_modality(preds.cpu(), y_te.cpu(), mods_te)
        best = max(best, acc["all"])
        if epoch % 20 == 0 or epoch == args.epochs:
            per_mod = "  ".join(f"{m}={acc[m]:.3f}" for m in sorted(acc) if m != "all")
            print(f"Epoch {epoch:3d} | attack acc all={acc['all']:.3f}  ({per_mod})")

    print(f"\n=== identity attack ===")
    print(f"final={acc['all']:.3f}  best={best:.3f}  majority-class={majority:.3f}")
    for m in sorted(acc):
        if m != "all":
            print(f"  {m}: {acc[m]:.3f}")


if __name__ == "__main__":
    main()
