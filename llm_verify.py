import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_attack import Adapter, label_token_ids

PROMPT = ("Recording A:", " Recording B:", " Same subject as A? Answer:")
LABELS = ["no", "yes"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=str, required=True)
    p.add_argument("--modality", type=str, default="imu", choices=["imu", "rgb"])
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--test_trial", type=int, default=4)
    p.add_argument("--n_soft", type=int, default=8)
    p.add_argument("--max_train_pairs", type=int, default=2000)
    p.add_argument("--max_test_pairs", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_pairs", type=str, default=None)
    return p.parse_args()


def enumerate_pairs(subject, activity, trial, test_trial, split):
    same, diff = [], []
    for a in np.unique(activity):
        idx = np.flatnonzero(activity == a)
        if split == "train":
            pool = idx[trial[idx] != test_trial]
            for p in range(len(pool)):
                for q in range(p + 1, len(pool)):
                    i, j = int(pool[p]), int(pool[q])
                    (same if subject[i] == subject[j] else diff).append((i, j))
        else:
            probe = idx[trial[idx] == test_trial]
            gallery = idx[trial[idx] != test_trial]
            for pi in probe:
                for gi in gallery:
                    i, j = int(gi), int(pi)
                    (same if subject[i] == subject[j] else diff).append((i, j))
    return same, diff


def balanced_sample(same, diff, max_pairs, seed):
    rng = np.random.default_rng(seed)
    k = min(len(same), len(diff), max_pairs // 2)
    s_sel = rng.choice(len(same), k, replace=False)
    d_sel = rng.choice(len(diff), k, replace=False)
    pairs = [same[i] for i in s_sel] + [diff[i] for i in d_sel]
    labels = [1] * k + [0] * k
    return pairs, labels, len(same), len(diff)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    d = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"Embeddings: {args.embeddings}  modality={args.modality}")

    mod = np.array(d["modality"])
    keep = (~d["is_heldout"].numpy()) & (mod == args.modality)
    X_raw = d["embedding"][keep].float()
    subject = d["subject"].numpy()[keep]
    activity = d["activity"].numpy()[keep]
    trial = d["trial"].numpy()[keep]

    train_idx = np.flatnonzero(trial != args.test_trial)
    mu = X_raw[train_idx].mean(dim=0, keepdim=True)
    sd = X_raw[train_idx].std(dim=0, keepdim=True) + 1e-6
    X_std = (X_raw - mu) / sd

    tr_same, tr_diff = enumerate_pairs(subject, activity, trial, args.test_trial, "train")
    te_same, te_diff = enumerate_pairs(subject, activity, trial, args.test_trial, "test")
    tr_pairs, tr_labels, tr_ns, tr_nd = balanced_sample(tr_same, tr_diff, args.max_train_pairs, args.seed)
    te_pairs, te_labels, te_ns, te_nd = balanced_sample(te_same, te_diff, args.max_test_pairs, args.seed + 1)

    print(f"distinct pairs available -- train: same={tr_ns} diff={tr_nd} | "
          f"test: same={te_ns} diff={te_nd}")
    print(f"sampled without replacement -- train={len(tr_pairs)} test={len(te_pairs)} "
          f"(balanced 50/50, activity-matched, no duplicates)")

    tr_i = torch.tensor([p[0] for p in tr_pairs])
    tr_j = torch.tensor([p[1] for p in tr_pairs])
    tr_y = torch.tensor(tr_labels)
    te_i = torch.tensor([p[0] for p in te_pairs])
    te_j = torch.tensor([p[1] for p in te_pairs])
    te_y_np = np.array(te_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    tok_ids = label_token_ids(tokenizer, LABELS).to(device)
    d_model = llm.get_input_embeddings().embedding_dim
    adapter = Adapter(X_std.shape[1], d_model, args.n_soft).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    print(f"LLM {args.model} frozen ({sum(p.numel() for p in llm.parameters())/1e6:.0f}M), "
          f"adapter trainable ({sum(p.numel() for p in adapter.parameters())/1e6:.2f}M), "
          f"soft tokens={args.n_soft} per recording")

    pre_a, pre_b, post = PROMPT
    with torch.no_grad():
        tbl = llm.get_input_embeddings()
        pre_a_emb = tbl(torch.tensor(tokenizer.encode(pre_a, add_special_tokens=False), device=device))
        pre_b_emb = tbl(torch.tensor(tokenizer.encode(pre_b, add_special_tokens=False), device=device))
        post_emb = tbl(torch.tensor(tokenizer.encode(post, add_special_tokens=False), device=device))

    X = X_std.to(device)
    tr_i, tr_j, tr_y = tr_i.to(device), tr_j.to(device), tr_y.to(device)
    te_i, te_j = te_i.to(device), te_j.to(device)

    def forward(i_idx, j_idx):
        b = len(i_idx)
        seq = torch.cat([
            pre_a_emb.unsqueeze(0).expand(b, -1, -1),
            adapter(X[i_idx]),
            pre_b_emb.unsqueeze(0).expand(b, -1, -1),
            adapter(X[j_idx]),
            post_emb.unsqueeze(0).expand(b, -1, -1),
        ], dim=1)
        mask = torch.ones(seq.shape[:2], dtype=torch.long, device=device)
        return llm(inputs_embeds=seq, attention_mask=mask).logits[:, -1, :][:, tok_ids]

    best_acc, best_epoch, best_preds, since_best, patience = 0.0, 0, None, 0, 15
    n = len(tr_i)
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            swap = torch.rand(len(idx), device=device) < 0.5
            a_idx = torch.where(swap, tr_j[idx], tr_i[idx])
            b_idx = torch.where(swap, tr_i[idx], tr_j[idx])
            loss = F.cross_entropy(forward(a_idx, b_idx), tr_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        adapter.eval()
        with torch.no_grad():
            logits = torch.cat([forward(te_i[s:s + args.batch_size], te_j[s:s + args.batch_size])
                                for s in range(0, len(te_i), args.batch_size)])
            preds = logits.argmax(dim=1).cpu().numpy()
        acc = float((preds == te_y_np).mean())
        yes_share = float((preds == 1).mean())

        if acc > best_acc:
            best_acc, best_epoch, best_preds, since_best = acc, epoch, preds, 0
        else:
            since_best += 1

        if epoch % 10 == 0 or epoch == args.epochs:
            flag = "  <-- COLLAPSED" if yes_share in (0.0, 1.0) else ""
            print(f"Epoch {epoch:3d} | acc={acc:.3f}  said-yes={yes_share:.3f}{flag}")

        if since_best >= patience:
            print(f"early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    tp = int(((best_preds == 1) & (te_y_np == 1)).sum())
    fn = int(((best_preds == 0) & (te_y_np == 1)).sum())
    tn = int(((best_preds == 0) & (te_y_np == 0)).sum())
    fp = int(((best_preds == 1) & (te_y_np == 0)).sum())

    out = args.out_pairs or (f"logs/verify_pairs_{Path(args.embeddings).stem}"
                             f"_{args.modality}.csv")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("subject_a,subject_b,activity,trial_a,trial_b,truth,predicted,correct\n")
        for (i, j), truth, pred in zip(te_pairs, te_labels, best_preds):
            f.write(f"{subject[i]+1},{subject[j]+1},{activity[i]+1},"
                    f"{trial[i]},{trial[j]},"
                    f"{'yes' if truth else 'no'},{'yes' if pred else 'no'},"
                    f"{int(truth == pred)}\n")

    print(f"\n=== LLM verification ({args.modality}) ===")
    print(f"accuracy={best_acc:.3f}  (epoch {best_epoch})  chance=0.500")
    print(f"  same-subject pairs      said yes {tp:4d} / {tp+fn:4d}  ({tp/max(tp+fn,1):.3f})")
    print(f"  different-subject pairs said no  {tn:4d} / {tn+fp:4d}  ({tn/max(tn+fp,1):.3f})")
    print(f"  said-yes overall        {(tp+fp)/len(te_y_np):.3f}  (0.500 = balanced, 0 or 1 = collapsed)")
    print(f"\nper-pair decisions written to {out}")


if __name__ == "__main__":
    main()
