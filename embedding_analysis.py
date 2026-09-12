import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exports", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--outdir", type=str, default="figures")
    p.add_argument("--modality", type=str, default="all", choices=["all", "imu", "rgb"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load(path, modality="all"):
    d = torch.load(path, map_location="cpu", weights_only=False)
    keep = np.ones(len(d["subject"]), dtype=bool)
    if modality != "all":
        keep = np.array([m == modality for m in d["modality"]])
    return {
        "emb": d["embedding"].numpy()[keep],
        "subject": d["subject"].numpy()[keep],
        "activity": d["activity"].numpy()[keep],
        "trial": d["trial"].numpy()[keep],
        "heldout": d["is_heldout"].numpy()[keep],
        "index": d["index"].numpy()[keep],
        "val_subject": d["val_subject"],
        "path": path,
    }


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def silhouettes(d):
    tr = ~d["heldout"]
    X = l2norm(d["emb"][tr])
    out = {}
    for key in ("subject", "activity"):
        lab = d[key][tr]
        out[key] = float(silhouette_score(X, lab, metric="cosine")) if len(set(lab)) > 1 else float("nan")
    return out


def verification(d, seed=0, n_pairs=20000):
    rng = np.random.default_rng(seed)
    tr = ~d["heldout"]
    X = l2norm(d["emb"][tr])
    subj = d["subject"][tr]
    act = d["activity"][tr]

    idx_by_act_subj = {}
    for a in np.unique(act):
        by_subj = {s: np.flatnonzero((act == a) & (subj == s)) for s in np.unique(subj[act == a])}
        by_subj = {s: idx for s, idx in by_subj.items() if len(idx) >= 1}
        if len(by_subj) >= 2:
            idx_by_act_subj[a] = by_subj

    same, diff = [], []
    activities = list(idx_by_act_subj)
    for _ in range(n_pairs // 2):
        a = rng.choice(activities)
        by_subj = idx_by_act_subj[a]
        subjects = list(by_subj)

        s = rng.choice(subjects)
        if len(by_subj[s]) >= 2:
            i, j = rng.choice(by_subj[s], 2, replace=False)
            same.append(float(X[i] @ X[j]))

        s1, s2 = rng.choice(subjects, 2, replace=False)
        i = rng.choice(by_subj[s1])
        j = rng.choice(by_subj[s2])
        diff.append(float(X[i] @ X[j]))

    scores = np.array(same + diff)
    labels = np.array([1] * len(same) + [0] * len(diff))
    auc = float(roc_auc_score(labels, scores))

    order = np.argsort(scores)
    s_sorted, l_sorted = scores[order], labels[order]
    n_pos, n_neg = l_sorted.sum(), len(l_sorted) - l_sorted.sum()
    fn = np.cumsum(l_sorted) / max(n_pos, 1)
    fp = 1.0 - np.cumsum(1 - l_sorted) / max(n_neg, 1)
    eer = float(np.min(np.maximum(fn, fp)))
    return {"auc": auc, "eer": eer, "n_same": len(same), "n_diff": len(diff)}


def unseen_separability(d):
    X = l2norm(d["emb"])
    held = d["heldout"]
    if held.sum() == 0:
        return {"self_sim": float("nan"), "cross_sim": float("nan"), "margin": float("nan")}

    held_c = X[held].mean(0)
    held_c /= np.linalg.norm(held_c) + 1e-12
    self_sim = float(np.mean(X[held] @ held_c))

    cross = []
    for s in np.unique(d["subject"][~held]):
        c = X[(~held) & (d["subject"] == s)].mean(0)
        c /= np.linalg.norm(c) + 1e-12
        cross.append(float(np.mean(X[held] @ c)))

    cross_sim = float(np.mean(cross))
    return {"self_sim": self_sim, "cross_sim": cross_sim, "margin": self_sim - cross_sim}


def representation_shift(a, b):
    common = np.intersect1d(a["index"], b["index"])
    ia = {v: k for k, v in enumerate(a["index"])}
    ib = {v: k for k, v in enumerate(b["index"])}
    Xa = l2norm(a["emb"][[ia[c] for c in common]])
    Xb = l2norm(b["emb"][[ib[c] for c in common]])
    sims = np.sum(Xa * Xb, axis=1)
    return {"mean_cosine": float(sims.mean()), "std": float(sims.std()), "n": len(common)}


def tsne_panel(datasets, labels, outdir, seed=0):
    fig, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 11), squeeze=False)
    for col, (d, name) in enumerate(zip(datasets, labels)):
        tr = ~d["heldout"]
        X = l2norm(d["emb"][tr])
        proj = TSNE(n_components=2, metric="cosine", init="pca",
                    random_state=seed, perplexity=30).fit_transform(X)

        for row, key in enumerate(("subject", "activity")):
            ax = axes[row][col]
            lab = d[key][tr]
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=lab, cmap="tab10" if key == "subject" else "tab20",
                            s=6, alpha=0.75)
            ax.set_title(f"{name} — coloured by {key}")
            ax.set_xticks([])
            ax.set_yticks([])
            if key == "subject":
                fig.colorbar(sc, ax=ax, fraction=0.046, label="subject")

    fig.tight_layout()
    path = Path(outdir) / "fig1_tsne.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def silhouette_scatter(rows, outdir):
    pts = [(r["silhouette_subject"], r["label"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 5))
    for x, name in pts:
        ax.scatter(x, 0, s=80)
        ax.annotate(name, (x, 0), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_xlabel("silhouette by subject (higher = identity more clustered)")
    ax.set_yticks([])
    ax.set_title("Identity structure in embedding space")
    fig.tight_layout()
    path = Path(outdir) / "fig2_silhouette.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    labels = args.labels or [Path(p).stem for p in args.exports]
    if len(labels) != len(args.exports):
        raise SystemExit("--labels must match --exports in length")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    datasets = [load(p, args.modality) for p in args.exports]

    rows = []
    for d, name in zip(datasets, labels):
        sil = silhouettes(d)
        ver = verification(d, args.seed)
        uns = unseen_separability(d)
        rows.append({
            "label": name,
            "n": len(d["subject"]),
            "silhouette_subject": sil["subject"],
            "silhouette_activity": sil["activity"],
            "verif_auc": ver["auc"],
            "verif_eer": ver["eer"],
            "unseen_self": uns["self_sim"],
            "unseen_cross": uns["cross_sim"],
            "unseen_margin": uns["margin"],
        })

    print(f"\n=== Table 2: embedding analysis (modality={args.modality}) ===")
    hdr = (f"{'model':<14}{'sil_subj':>10}{'sil_act':>10}{'verif_AUC':>11}"
           f"{'verif_EER':>11}{'unseen_margin':>15}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<14}{r['silhouette_subject']:>10.3f}{r['silhouette_activity']:>10.3f}"
              f"{r['verif_auc']:>11.3f}{r['verif_eer']:>11.3f}{r['unseen_margin']:>15.3f}")

    if len(datasets) == 2:
        shift = representation_shift(datasets[0], datasets[1])
        print(f"\nrepresentation shift {labels[0]} vs {labels[1]}: "
              f"cosine {shift['mean_cosine']:.3f} +/- {shift['std']:.3f} over {shift['n']} samples")

    f1 = tsne_panel(datasets, labels, args.outdir, args.seed)
    f2 = silhouette_scatter(rows, args.outdir)
    print(f"\nwrote {f1}")
    print(f"wrote {f2}")


if __name__ == "__main__":
    main()
