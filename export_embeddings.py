import argparse
from pathlib import Path

import torch

from ppf.data.dataloader import UTDMHADDataset, compute_normalization_stats
from attack import build_frozen, trial_of


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--root", type=str, default="UTD-MHAD")
    p.add_argument("--val_subject", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


@torch.no_grad()
def embed_indices(dataset, indices, encoders, projectors, device, batch_size=32):
    by_mod = {}
    for i in indices:
        by_mod.setdefault(dataset.samples[i]["modality"], []).append(i)

    embs, order = [], []
    for modality, idxs in by_mod.items():
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start:start + batch_size]
            x = torch.stack([dataset[i]["input_tensor"] for i in chunk]).to(device)
            embs.append(projectors[modality](encoders[modality](x)).cpu())
            order.extend(chunk)

    emb = torch.cat(embs)
    perm = sorted(range(len(order)), key=lambda k: order[k])
    return emb[perm], [order[k] for k in perm]


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
    print(f"normalisation from {len(train_idx)} training-subject samples "
          f"(val_subject={args.val_subject} excluded)")

    encoders, projectors = build_frozen(ckpt, device)

    all_idx = list(range(len(dataset.samples)))
    print(f"Embedding all {len(all_idx)} samples...")
    emb, order = embed_indices(dataset, all_idx, encoders, projectors, device, args.batch_size)

    meta = [dataset.samples[i] for i in order]
    payload = {
        "embedding": emb,
        "subject": torch.tensor([s["subject"] - 1 for s in meta]),
        "activity": torch.tensor([s["activity"] - 1 for s in meta]),
        "trial": torch.tensor([trial_of(s) for s in meta]),
        "modality": [s["modality"] for s in meta],
        "is_heldout": torch.tensor([s["subject"] == args.val_subject for s in meta]),
        "index": torch.tensor(order),
        "val_subject": args.val_subject,
        "config": ckpt["config"],
        "checkpoint": args.checkpoint,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    n_held = int(payload["is_heldout"].sum())
    print(f"\nSaved {args.out}")
    print(f"  embeddings {tuple(emb.shape)}")
    print(f"  subjects   {sorted(set(payload['subject'].tolist()))}")
    print(f"  activities {len(set(payload['activity'].tolist()))}")
    print(f"  trials     {sorted(set(payload['trial'].tolist()))}")
    print(f"  modalities {sorted(set(payload['modality']))}")
    print(f"  held-out   {n_held} samples, {len(all_idx) - n_held} training")


if __name__ == "__main__":
    main()
