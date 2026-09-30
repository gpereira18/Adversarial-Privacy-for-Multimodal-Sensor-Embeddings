import argparse
from collections import Counter

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
    p.add_argument("--n_train_pairs", type=int, default=2000)
    p.add_argument("--n_test_pairs", type=int, default=600)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def sample_pairs(subject, activity, trial, test_trial, n_pairs, seed, split):
    rng = np.random.default_rng(seed)
    by_act_subj = {}
    for a in np.unique(activity):
        act_mask = activity == a
        by_subj = {s: np.flatnonzero(act_mask & (subject == s)) for s in np.unique(subject[act_mask])}
        by_act_subj[a] = by_subj

    pairs, labels = [], []
    activities = list(by_act_subj)
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        a = rng.choice(activities)
        by_subj = by_act_subj[a]
        subjects = list(by_subj)
        if len(subjects) < 2:
            continue

        want_same = len(pairs) % 2 == 0
        if want_same:
            s = rng.choice(subjects)
            pool = by_subj[s]
            if len(pool) < 2:
                continue
            i, j = rng.choice(pool, 2, replace=False)
        else:
            s1, s2 = rng.choice(subjects, 2, replace=False)
            i, j = rng.choice(by_subj[s1]), rng.choice(by_subj[s2])

        ti, tj = trial[i], trial[j]
        if split == "train":
            if ti == test_trial or tj == test_trial:
                continue
        else:
            is_probe = (ti == test_trial) != (tj == test_trial)
            if not is_probe:
                continue

        pairs.append((i, j))
        labels.append(int(want_same))

    return pairs, labels


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    d = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"Embeddings: {args.embeddings}  (modality={args.modality}, single modality only "
          f"-- mixing modalities would let the LLM key on modality instead of identity)")

    mod = np.array(d["modality"])
    keep = (~d["is_heldout"].numpy()) & (mod == args.modality)
    X_full = d["embedding"][keep].float()
    subject = d["subject"].numpy()[keep]
    activity = d["activity"].numpy()[keep]
    trial = d["trial"].numpy()[keep]

    train_idx = (trial != args.test_trial).nonzero()[0]
    mu = X_full[train_idx].mean(dim=0, keepdim=True)
    sd = X_full[train_idx].std(dim=0, keepdim=True) + 1e-6
    X_full = (X_full - mu) / sd

    tr_pairs, tr_labels = sample_pairs(subject, activity, trial, args.test_trial,
                                        args.n_train_pairs, args.seed, "train")
    te_pairs, te_labels = sample_pairs(subject, activity, trial, args.test_trial,
                                        args.n_test_pairs, args.seed + 1, "test")
    print(f"train pairs={len(tr_pairs)} (same={sum(tr_labels)})  "
          f"test pairs={len(te_pairs)} (same={sum(te_labels)})")
    if len(tr_pairs) < args.n_train_pairs * 0.5 or len(te_pairs) < args.n_test_pairs * 0.5:
        print("WARNING: sampling fell well short of the requested pair counts -- "
              "too few subjects/trials per activity for this modality/fold")

    tr_i = torch.tensor([p[0] for p in tr_pairs])
    tr_j = torch.tensor([p[1] for p in tr_pairs])
    tr_y = torch.tensor(tr_labels)
    te_i = torch.tensor([p[0] for p in te_pairs])
    te_j = torch.tensor([p[1] for p in te_pairs])
    te_y = torch.tensor(te_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    tok_ids = label_token_ids(tokenizer, LABELS).to(device)
    print(f"label strings: {' '.join(LABELS)}")

    d_model = llm.get_input_embeddings().embedding_dim
    adapter = Adapter(X_full.shape[1], d_model, args.n_soft).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    print(f"LLM {args.model} frozen ({sum(p.numel() for p in llm.parameters())/1e6:.0f}M), "
          f"adapter trainable ({sum(p.numel() for p in adapter.parameters())/1e6:.2f}M), "
          f"soft tokens={args.n_soft} per recording")

    pre_a, pre_b, post = PROMPT
    with torch.no_grad():
        emb_table = llm.get_input_embeddings()
        pre_a_emb = emb_table(torch.tensor(tokenizer.encode(pre_a, add_special_tokens=False), device=device))
        pre_b_emb = emb_table(torch.tensor(tokenizer.encode(pre_b, add_special_tokens=False), device=device))
        post_emb = emb_table(torch.tensor(tokenizer.encode(post, add_special_tokens=False), device=device))

    X = X_full.to(device)
    tr_i, tr_j, tr_y = tr_i.to(device), tr_j.to(device), tr_y.to(device)
    te_i, te_j, te_y = te_i.to(device), te_j.to(device), te_y.to(device)

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
        out = llm(inputs_embeds=seq, attention_mask=mask).logits[:, -1, :]
        return out[:, tok_ids]

    best, best_epoch, since_best, patience = 0.0, 0, 0, 15
    n = len(tr_i)
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            loss = F.cross_entropy(forward(tr_i[idx], tr_j[idx]), tr_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        adapter.eval()
        with torch.no_grad():
            preds = torch.cat([forward(te_i[s:s + args.batch_size], te_j[s:s + args.batch_size]).argmax(dim=1)
                               for s in range(0, len(te_i), args.batch_size)])
            acc = (preds == te_y).float().mean().item()

        if acc > best:
            best, best_epoch, since_best = acc, epoch, 0
        else:
            since_best += 1

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d} | verify acc={acc:.3f}")

        if since_best >= patience:
            print(f"early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    majority = max(Counter(te_labels).values()) / len(te_labels)
    print(f"\n=== LLM verification ({args.modality}) ===")
    print(f"best={best:.3f}  (epoch {best_epoch})  chance=0.500  majority-class={majority:.3f}")


if __name__ == "__main__":
    main()
