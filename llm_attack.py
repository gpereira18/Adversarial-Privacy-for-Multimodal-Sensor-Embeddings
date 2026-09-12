import argparse
import string
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from attack import accuracy_by_modality

PROMPTS = {
    "subject": ("Recording:", " Which subject performed this? Answer:"),
    "activity": ("Recording:", " Which activity is this? Answer:"),
}


class Adapter(nn.Module):
    def __init__(self, in_dim, d_model, n_soft):
        super().__init__()
        self.n_soft = n_soft
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_soft * d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.net(x).view(-1, self.n_soft, self.d_model)
        return self.norm(h)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=str, required=True)
    p.add_argument("--target", type=str, default="subject", choices=["subject", "activity"])
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--test_trial", type=int, default=4)
    p.add_argument("--n_soft", type=int, default=8)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def label_strings(target, n_classes):
    if target == "subject":
        pool = [str(i) for i in range(1, 10)]
    else:
        pool = list(string.ascii_uppercase) + [str(i) for i in range(10)]
    if n_classes > len(pool):
        raise SystemExit(f"need {n_classes} label strings, pool has {len(pool)}")
    return pool[:n_classes]


def label_token_ids(tokenizer, labels):
    ids = []
    for s in labels:
        toks = tokenizer.encode(" " + s, add_special_tokens=False)
        if not toks:
            raise SystemExit(f"label {s!r} tokenised to nothing")
        ids.append(toks[-1])
    if len(set(ids)) != len(ids):
        dupes = [s for s, i in zip(labels, ids) if ids.count(i) > 1]
        raise SystemExit(f"label tokens collide for {dupes} — pick different label strings")
    return torch.tensor(ids)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    d = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"Embeddings: {args.embeddings}")
    print(f"From checkpoint: {d.get('checkpoint')}")

    keep = ~d["is_heldout"]
    X = d["embedding"][keep]
    y = d[args.target][keep]
    trial = d["trial"][keep]
    mods = [m for m, k in zip(d["modality"], keep.tolist()) if k]

    classes = sorted(set(y.tolist()))
    remap = {c: i for i, c in enumerate(classes)}
    y = torch.tensor([remap[v] for v in y.tolist()])
    n_classes = len(classes)

    tr = (trial != args.test_trial).nonzero(as_tuple=True)[0]
    te = (trial == args.test_trial).nonzero(as_tuple=True)[0]
    mods_te = [mods[i] for i in te.tolist()]
    y_te_cpu = y[te].clone()

    raw_norm = X.norm(dim=1).mean().item()
    mu = X[tr].mean(dim=0, keepdim=True)
    sd = X[tr].std(dim=0, keepdim=True) + 1e-6
    X = (X - mu) / sd
    print(f"standardised attacker inputs: raw mean-norm={raw_norm:.1f} -> "
          f"standardised mean-norm={X.norm(dim=1).mean().item():.1f}")

    majority = max(Counter(y[te].tolist()).values()) / len(te)
    print(f"target={args.target}  classes={n_classes}  train={len(tr)}  test={len(te)}  "
          f"majority-class={majority:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    labels = label_strings(args.target, n_classes)
    tok_ids = label_token_ids(tokenizer, labels).to(device)
    print(f"label strings: {' '.join(labels)}")

    d_model = llm.get_input_embeddings().embedding_dim
    adapter = Adapter(X.shape[1], d_model, args.n_soft).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    print(f"LLM {args.model} frozen ({sum(p.numel() for p in llm.parameters())/1e6:.0f}M), "
          f"adapter trainable ({sum(p.numel() for p in adapter.parameters())/1e6:.2f}M), "
          f"soft tokens={args.n_soft}")

    pre_text, post_text = PROMPTS[args.target]
    pre_ids = tokenizer.encode(pre_text, add_special_tokens=False)
    post_ids = tokenizer.encode(post_text, add_special_tokens=False)
    with torch.no_grad():
        emb_table = llm.get_input_embeddings()
        pre_emb = emb_table(torch.tensor(pre_ids, device=device))
        post_emb = emb_table(torch.tensor(post_ids, device=device))

    X, y = X.to(device), y.to(device)
    tr, te = tr.to(device), te.to(device)

    def forward(idx):
        b = len(idx)
        seq = torch.cat([
            pre_emb.unsqueeze(0).expand(b, -1, -1),
            adapter(X[idx]),
            post_emb.unsqueeze(0).expand(b, -1, -1),
        ], dim=1)
        mask = torch.ones(seq.shape[:2], dtype=torch.long, device=device)
        out = llm(inputs_embeds=seq, attention_mask=mask).logits[:, -1, :]
        return out[:, tok_ids]

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        perm = tr[torch.randperm(len(tr))]
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start:start + args.batch_size]
            loss = F.cross_entropy(forward(idx), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        adapter.eval()
        with torch.no_grad():
            preds = torch.cat([forward(te[i:i + args.batch_size]).argmax(dim=1)
                               for i in range(0, len(te), args.batch_size)])
            acc = accuracy_by_modality(preds.cpu(), y_te_cpu, mods_te)
        best = max(best, acc["all"])
        if epoch % 10 == 0 or epoch == args.epochs:
            per_mod = "  ".join(f"{m}={acc[m]:.3f}" for m in sorted(acc) if m != "all")
            print(f"Epoch {epoch:3d} | llm acc all={acc['all']:.3f}  ({per_mod})")

    print(f"\n=== LLM {args.target} attack ===")
    print(f"final={acc['all']:.3f}  best={best:.3f}  majority-class={majority:.3f}")
    for m in sorted(acc):
        if m != "all":
            print(f"  {m}: {acc[m]:.3f}")


if __name__ == "__main__":
    main()
