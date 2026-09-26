"""
train_synth.py -- Section 7.2 on the server, at the full budget.

The authoring-environment run used 3000 steps, which left wide models
UNDERTRAINED and made the width sweep uninterpretable (wider models carried
*higher* loss, the reverse of the usual regime). At 15000 steps the regime is
the intended one and the width claim can actually be tested.

Emits the same .npz schema as cifar/train.py, so one analyze.py serves both.

    python train_synth.py --classes 6 --width 64 --seed 0 --out runs_synth/
"""

import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


def make_data(c, n, seed, sigma=3.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, sigma, size=(n, 2))
    th = np.arctan2(X[:, 1], X[:, 0]) % (2 * np.pi)
    y = (np.floor(th / (np.pi / c)).astype(np.int64)) % c
    return X.astype(np.float32), y


class Embed(nn.Module):
    def __init__(self, w, M):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, w), nn.LeakyReLU(0.01),
            nn.Linear(w, w), nn.LeakyReLU(0.01),
            nn.Linear(w, w), nn.LeakyReLU(0.01),
            nn.Linear(w, M))

    def forward(self, x):
        return self.net(x)


class Unembed(nn.Module):
    def __init__(self, C, M):
        super().__init__()
        self.G = nn.Parameter(torch.randn(C, M))

    def forward(self):
        return self.G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', type=int, default=6)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--dim', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smooth', type=float, default=0.0)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--out', default='runs_synth')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.steps = 50
    os.makedirs(a.out, exist_ok=True)

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, y = make_data(a.classes, a.n, seed=0)          # data fixed across seeds
    Xt = torch.tensor(X, device=dev); yt = torch.tensor(y, device=dev)

    emb = Embed(a.width, a.dim).to(dev)
    une = Unembed(a.classes, a.dim).to(dev)
    opt = torch.optim.Adam(list(emb.parameters()) + list(une.parameters()),
                           lr=a.lr)
    g = torch.Generator(device='cpu').manual_seed(a.seed + 999)

    for t in range(1, a.steps + 1):
        idx = torch.randint(0, a.n, (a.bs,), generator=g).to(dev)
        logits = emb(Xt[idx]) @ une().T
        loss = Fn.cross_entropy(logits, yt[idx], label_smoothing=a.smooth)
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 5000 == 0:
            print(f"  step {t:>6}  loss {loss.item():.4f}", flush=True)

    emb.eval(); une.eval()
    with torch.no_grad():
        f = emb(Xt); G = une()
        lg = f @ G.T
        lp = lg - torch.logsumexp(lg, 1, keepdim=True)
        f, G, lp = f.cpu().numpy(), G.cpu().numpy(), lp.cpu().numpy()
    acc = float((lp.argmax(1) == y).mean())
    tr_loss = float(-lp[np.arange(len(y)), y].mean())

    tag = f"c{a.classes}_w{a.width}_d{a.dim}_s{a.seed}_sm{a.smooth}"
    np.savez_compressed(os.path.join(a.out, tag + '.npz'),
                        f=f.astype('f4'), g=G.astype('f4'),
                        logp=lp.astype('f4'), y=y.astype('i8'),
                        acc=acc, loss=tr_loss, dim=a.dim, seed=a.seed,
                        smooth=a.smooth, width=a.width, nclass=a.classes)
    print(json.dumps(dict(tag=tag, acc=acc, loss=tr_loss)), flush=True)


if __name__ == '__main__':
    main()
