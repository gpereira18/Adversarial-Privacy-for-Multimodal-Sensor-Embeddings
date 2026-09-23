import argparse
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exports", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--outdir", type=str, default="figures")
    p.add_argument("--modality", type=str, default="imu", choices=["all", "imu", "rgb"])
    p.add_argument("--activities", type=str, default="all",
                   help="'all' or comma-separated 1-indexed activity ids")
    p.add_argument("--per_activity", action="store_true",
                   help="print the full per-activity breakdown, not just the summary")
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


def activity_similarities(d, activities):
    rows, same_pool, diff_pool = [], [], []
    for a in activities:
        m = d["activity"] == a
        subj = d["subject"][m]
        if len(subj) < 2:
            continue
        X = l2norm(d["emb"][m])
        sim = X @ X.T
        i, j = np.triu_indices(len(subj), k=1)
        s = sim[i, j]
        same = subj[i] == subj[j]
        if same.sum() == 0 or (~same).sum() == 0:
            continue

        same_s, diff_s = s[same], s[~same]
        same_pool.append(same_s)
        diff_pool.append(diff_s)
        rows.append({
            "activity": int(a) + 1,
            "n": int(len(subj)),
            "same": float(same_s.mean()),
            "diff": float(diff_s.mean()),
            "gap": float(same_s.mean() - diff_s.mean()),
        })

    return rows, np.concatenate(same_pool), np.concatenate(diff_pool)


def summarise(rows, same_pool, diff_pool):
    gaps = [r["gap"] for r in rows]
    return {
        "n_act": len(rows),
        "same": float(same_pool.mean()),
        "diff": float(diff_pool.mean()),
        "gap_pooled": float(same_pool.mean() - diff_pool.mean()),
        "gap_mean": statistics.mean(gaps) if gaps else float("nan"),
        "gap_std": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
        "n_same": len(same_pool),
        "n_diff": len(diff_pool),
    }


def plot_distributions(pools, labels, outdir, modality):
    n = len(pools)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows),
                             squeeze=False, sharex=True)
    flat = [ax for row in axes for ax in row]

    lo = min(min(s.min(), d.min()) for s, d in pools)
    bins = np.linspace(lo, 1.0, 70)

    for ax, (same, diff), name in zip(flat, pools, labels):
        ax.hist(diff, bins=bins, alpha=0.55, density=True, label="different subject")
        ax.hist(same, bins=bins, alpha=0.55, density=True, label="same subject")
        ax.axvline(diff.mean(), ls="--", lw=1.2, color="tab:blue")
        ax.axvline(same.mean(), ls="--", lw=1.2, color="tab:orange")
        ax.set_title(f"{name}\ngap = {same.mean() - diff.mean():.3f}", fontsize=10)
        ax.set_xlabel("cosine similarity")
        ax.set_yticks([])
        ax.legend(fontsize=8)

    for ax in flat[n:]:
        ax.axis("off")

    fig.suptitle(f"Same- vs different-subject similarity ({modality}, all activities)",
                 fontsize=12)
    fig.tight_layout()
    path = Path(outdir) / f"similarity_distributions_{modality}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gap_scatter(rows_a, rows_b, label_a, label_b, outdir, modality):
    ga = {r["activity"]: r["gap"] for r in rows_a}
    gb = {r["activity"]: r["gap"] for r in rows_b}
    common = sorted(set(ga) & set(gb))
    x = [ga[a] for a in common]
    y = [gb[a] for a in common]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lim = max(max(x + y) * 1.15, 0.01)
    lo = min(0.0, min(x + y) * 1.15)
    ax.plot([lo, lim], [lo, lim], ls="--", c="gray", lw=1, label="no change")
    ax.scatter(x, y, s=70, alpha=0.8, zorder=3)
    for a, xi, yi in zip(common, x, y):
        ax.annotate(str(a), (xi, yi), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlim(lo, lim)
    ax.set_ylim(lo, lim)
    ax.set_xlabel(f"identity gap — {label_a}")
    ax.set_ylabel(f"identity gap — {label_b}")
    ax.set_title(f"Per-activity identity gap ({modality})\n"
                 f"{sum(1 for xi, yi in zip(x, y) if yi < xi)}/{len(common)} "
                 f"activities below the diagonal", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = Path(outdir) / f"gap_scatter_{modality}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    labels = args.labels or [Path(p).stem for p in args.exports]
    if len(labels) != len(args.exports):
        raise SystemExit("--labels must match --exports in length")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    raw = [load(p, args.modality) for p in args.exports]

    if args.activities == "all":
        activities = sorted(set(raw[0]["activity"].tolist()))
    else:
        activities = [int(a) - 1 for a in args.activities.split(",")]

    results = [activity_similarities(d, activities) for d in raw]
    summaries = [summarise(rows, sp, dp) for rows, sp, dp in results]

    print(f"\n=== identity gap over {len(activities)} activities (modality={args.modality}) ===")
    hdr = (f"{'model':<22}{'n_act':>7}{'same':>9}{'diff':>9}"
           f"{'gap':>9}{'gap_sd':>9}{'pairs':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name, s in zip(labels, summaries):
        print(f"{name:<22}{s['n_act']:>7}{s['same']:>9.3f}{s['diff']:>9.3f}"
              f"{s['gap_mean']:>9.3f}{s['gap_std']:>9.3f}"
              f"{s['n_same'] + s['n_diff']:>10}")

    if args.per_activity:
        for (rows, _, _), name in zip(results, labels):
            print(f"\n--- {name} ---")
            print(f"{'act':>5}{'n':>5}{'same':>9}{'diff':>9}{'gap':>9}")
            for r in rows:
                print(f"{r['activity']:>5}{r['n']:>5}{r['same']:>9.3f}"
                      f"{r['diff']:>9.3f}{r['gap']:>9.3f}")

    pools = [(sp, dp) for _, sp, dp in results]
    print()
    print(f"wrote {plot_distributions(pools, labels, args.outdir, args.modality)}")

    if len(raw) == 2:
        path = plot_gap_scatter(results[0][0], results[1][0],
                                labels[0], labels[1], args.outdir, args.modality)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
