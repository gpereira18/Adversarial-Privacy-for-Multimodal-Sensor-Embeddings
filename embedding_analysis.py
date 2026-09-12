import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exports", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--outdir", type=str, default="figures")
    p.add_argument("--modality", type=str, default="imu", choices=["all", "imu", "rgb"])
    p.add_argument("--activity", type=int, default=1)
    p.add_argument("--subjects", type=str, default=None,
                    help="comma-separated 1-indexed subject ids, e.g. '1,2', or 'all'. Default: first 2 present.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load(path, modality):
    d = torch.load(path, map_location="cpu", weights_only=False)
    keep = np.ones(len(d["subject"]), dtype=bool)
    if modality != "all":
        keep = np.array([m == modality for m in d["modality"]])
    keep &= ~d["is_heldout"].numpy()
    return {
        "emb": d["embedding"].numpy()[keep],
        "subject": d["subject"].numpy()[keep],
        "activity": d["activity"].numpy()[keep],
    }


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def filter_activity_subjects(d, activity, subjects):
    mask = d["activity"] == activity
    if subjects is not None:
        mask &= np.isin(d["subject"], subjects)
    return d["emb"][mask], d["subject"][mask]


def cosine_gap(emb, subj):
    X = l2norm(emb)
    sim = X @ X.T
    n = len(subj)
    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (same if subj[i] == subj[j] else diff).append(sim[i, j])
    same_mean = float(np.mean(same)) if same else float("nan")
    diff_mean = float(np.mean(diff)) if diff else float("nan")
    return same_mean, diff_mean, same_mean - diff_mean


def tsne_plot(datasets, labels, subjects, outdir, seed=0):
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5.5), squeeze=False)
    axes = axes[0]
    for ax, (emb, subj), name in zip(axes, datasets, labels):
        n = len(subj)
        perp = max(2, min(30, (n - 1) // 3))
        proj = TSNE(n_components=2, metric="cosine", init="pca",
                    random_state=seed, perplexity=perp).fit_transform(l2norm(emb))
        for s in subjects:
            m = subj == s
            if m.any():
                ax.scatter(proj[m, 0], proj[m, 1], label=f"subject {s + 1}", s=70, alpha=0.85)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(outdir) / "tsne_identity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    labels = args.labels or [Path(p).stem for p in args.exports]
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    activity = args.activity - 1
    raw = [load(p, args.modality) for p in args.exports]

    if args.subjects == "all":
        subjects = sorted(set(raw[0]["subject"][raw[0]["activity"] == activity].tolist()))
    elif args.subjects:
        subjects = [int(s) - 1 for s in args.subjects.split(",")]
    else:
        present = sorted(set(raw[0]["subject"][raw[0]["activity"] == activity].tolist()))
        subjects = present[:2]

    print(f"activity={activity + 1}  subjects={[s + 1 for s in subjects]}  modality={args.modality}")

    filtered = [filter_activity_subjects(d, activity, subjects) for d in raw]

    print(f"\n{'model':<20}{'n':>5}{'same-subj cos':>16}{'diff-subj cos':>16}{'gap':>10}")
    for (emb, subj), name in zip(filtered, labels):
        same, diff, gap = cosine_gap(emb, subj)
        print(f"{name:<20}{len(subj):>5}{same:>16.3f}{diff:>16.3f}{gap:>10.3f}")

    path = tsne_plot(filtered, labels, subjects, args.outdir, args.seed)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
