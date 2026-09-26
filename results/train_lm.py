"""
train_lm.py -- Section 7.4, the language-model check.

The model class is motivated by autoregressive LMs, where diversity is easy
(|V| ~ 10^4 against M ~ 10^2-10^3) but confidence is extreme -- only a few
next tokens are ever plausible. That is exactly the regime where
Cor.~15 says likelihood agreement should carry least representational
information, so it is the sharpest test of the paper's practical claim.

Small decoder-only transformer with TIED embeddings, so the unembedding matrix
g is the input embedding and the model is literally Eq. (1) of the paper.
Emits the shared .npz schema.

    python train_lm.py --text corpus.txt --seed 0 --out runs_lm/

With no --text it falls back to a synthetic 2nd-order Markov corpus, so the
script always runs; use a real corpus for the paper.
"""

import argparse, os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


def load_corpus(path, n_fallback=400000, seed=0):
    if path and os.path.exists(path):
        return open(path, 'r', encoding='utf-8', errors='ignore').read()
    rng = np.random.default_rng(seed)
    V = 96
    T = rng.dirichlet(np.ones(V) * 0.3, size=(V,))
    out, prev = [], 0
    for _ in range(n_fallback):
        prev = int(rng.choice(V, p=T[prev]))
        out.append(prev)
    return ''.join(chr(32 + c) for c in out)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.att(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TiedLM(nn.Module):
    def __init__(self, V, d, nl, nh, ctx):
        super().__init__()
        self.emb = nn.Embedding(V, d)          # tied: also the unembedding
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.ctx = ctx

    def features(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.full((T, T), float('-inf'),
                                     device=idx.device), 1)
        for b in self.blocks:
            x = b(x, mask)
        return self.ln(x)                      # (B, T, d) = f(context)

    def forward(self, idx):
        return self.features(idx) @ self.emb.weight.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dim', type=int, default=64)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--ctx', type=int, default=128)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--n_eval', type=int, default=4000)
    ap.add_argument('--out', default='runs_lm')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.steps = 50
    os.makedirs(a.out, exist_ok=True)

    text = load_corpus(a.text, seed=0)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    V = len(chars)
    ntr = int(0.9 * len(data))
    tr, te = data[:ntr], data[ntr:]
    print(f"vocab {V}  train {len(tr)}  test {len(te)}  "
          f"(diversity needs V > M = {a.dim}: "
          f"{'OK' if V > a.dim else 'FAILS'})", flush=True)

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = TiedLM(V, a.dim, a.layers, a.heads, a.ctx).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(a.seed + 7)

    def batch(src, bs):
        i = torch.randint(0, len(src) - a.ctx - 1, (bs,), generator=g)
        x = torch.stack([torch.from_numpy(src[j:j + a.ctx]) for j in i])
        y = torch.stack([torch.from_numpy(src[j + 1:j + a.ctx + 1]) for j in i])
        return x.to(dev), y.to(dev)

    m.train()
    for t in range(1, a.steps + 1):
        x, y = batch(tr, a.bs)
        loss = Fn.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if t % 2000 == 0:
            print(f"  step {t:>6}  loss {loss.item():.4f}  "
                  f"ppl {math.exp(loss.item()):.2f}", flush=True)

    # ---- dump last-position features and their targets as (f, g, logp) ----
    m.eval()
    F_, LP, Y = [], [], []
    with torch.no_grad():
        W = m.emb.weight                               # (V, d) unembedding
        need = a.n_eval
        while need > 0:
            b = min(64, need)
            i = torch.randint(0, len(te) - a.ctx - 1, (b,), generator=g)
            x = torch.stack([torch.from_numpy(te[j:j + a.ctx]) for j in i]).to(dev)
            yt = torch.stack([torch.from_numpy(te[j + a.ctx:j + a.ctx + 1])
                              for j in i]).squeeze(1).to(dev)
            f = m.features(x)[:, -1, :]                # (b, d)
            lg = f @ W.T
            LP.append((lg - torch.logsumexp(lg, 1, keepdim=True)).cpu().numpy())
            F_.append(f.cpu().numpy()); Y.append(yt.cpu().numpy())
            need -= b
    f = np.concatenate(F_); lp = np.concatenate(LP); y = np.concatenate(Y)
    acc = float((lp.argmax(1) == y).mean())
    nll = float(-lp[np.arange(len(y)), y].mean())

    tag = f"lm_d{a.dim}_s{a.seed}"
    np.savez_compressed(os.path.join(a.out, tag + '.npz'),
                        f=f.astype('f4'),
                        g=m.emb.weight.detach().cpu().numpy().astype('f4'),
                        logp=lp.astype('f4'), y=y.astype('i8'),
                        acc=acc, loss=nll, dim=a.dim, seed=a.seed,
                        smooth=0.0, width=0, nclass=V)
    print(json.dumps(dict(tag=tag, acc=acc, nll=nll, ppl=math.exp(nll))),
          flush=True)


if __name__ == '__main__':
    main()
