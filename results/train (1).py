"""
train.py -- CIFAR-10 models for experiments C1-C4.

Matches the protocol of the prior work: ResNet18 embedding network with the
final layer producing an M-dimensional representation; unembedding network is
three fully connected LeakyReLU layers of width 128 followed by an output layer
of size M. Adam, batch 32, 20k steps.

Each run dumps an .npz consumed by analyze.py, so training happens once and the
analysis is cheap and re-runnable.

    python train.py --dim 2 --seed 0 --smooth 0.0 --out runs/

NOTE: this file could not be executed in the authoring environment (no GPU, no
dataset access). The measurement layer in measures.py IS tested
(test_measures.py). Run --smoke first on your server; it does 50 steps on one
batch and writes a dummy npz, which catches shape and dtype errors in about a
minute before you commit 27 GPU-hours.
"""

import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ------------------------------------------------------------- ResNet18 -----
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, planes, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(inp, planes, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(planes)
        self.sc = nn.Sequential()
        if stride != 1 or inp != planes:
            self.sc = nn.Sequential(
                nn.Conv2d(inp, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes))

    def forward(self, x):
        o = Fn.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return Fn.relu(o + self.sc(x))


class ResNet18(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.inp = 64
        self.c1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        self.l1 = self._layer(64, 2, 1)
        self.l2 = self._layer(128, 2, 2)
        self.l3 = self._layer(256, 2, 2)
        self.l4 = self._layer(512, 2, 2)
        self.fc = nn.Linear(512, M)

    def _layer(self, planes, n, stride):
        layers, strides = [], [stride] + [1] * (n - 1)
        for s in strides:
            layers.append(BasicBlock(self.inp, planes, s))
            self.inp = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        o = Fn.relu(self.b1(self.c1(x)))
        o = self.l4(self.l3(self.l2(self.l1(o))))
        o = Fn.adaptive_avg_pool2d(o, 1).flatten(1)
        return self.fc(o)


class Unembed(nn.Module):
    """Three FC LeakyReLU layers of width 128, then an output layer of size M."""

    def __init__(self, C, M, w=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, w), nn.LeakyReLU(0.01),
            nn.Linear(w, w), nn.LeakyReLU(0.01),
            nn.Linear(w, w), nn.LeakyReLU(0.01),
            nn.Linear(w, M))
        self.register_buffer('eye', torch.eye(C))

    def forward(self):
        return self.net(self.eye)              # (C, M)


# ----------------------------------------------------------------- train ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smooth', type=float, default=0.0)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--data', default='./data')
    ap.add_argument('--out', default='runs')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.steps = 50
    os.makedirs(a.out, exist_ok=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616))
    tr_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), norm])
    te_tf = transforms.Compose([transforms.ToTensor(), norm])
    tr = datasets.CIFAR10(a.data, True, tr_tf, download=True)
    te = datasets.CIFAR10(a.data, False, te_tf, download=True)
    trl = DataLoader(tr, a.bs, shuffle=True, num_workers=4, drop_last=True)
    tel = DataLoader(te, 500, shuffle=False, num_workers=4)

    C = 10
    emb = ResNet18(a.dim).to(dev)
    une = Unembed(C, a.dim).to(dev)
    opt = torch.optim.Adam(list(emb.parameters()) + list(une.parameters()),
                           lr=a.lr)

    step, it = 0, iter(trl)
    emb.train(); une.train()
    while step < a.steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(trl); x, y = next(it)
        x, y = x.to(dev), y.to(dev)
        f = emb(x); g = une()
        logits = f @ g.T
        loss = Fn.cross_entropy(logits, y, label_smoothing=a.smooth)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 2000 == 0:
            print(f"  step {step:>6}  loss {loss.item():.4f}", flush=True)

    # ------------------------------------------------------------- dump ----
    emb.eval(); une.eval()
    F_, LP, Y = [], [], []
    with torch.no_grad():
        g = une()
        for x, y in tel:
            f = emb(x.to(dev))
            lg = f @ g.T
            LP.append((lg - torch.logsumexp(lg, 1, keepdim=True)).cpu().numpy())
            F_.append(f.cpu().numpy()); Y.append(y.numpy())
    f = np.concatenate(F_); lp = np.concatenate(LP); y = np.concatenate(Y)
    acc = float((lp.argmax(1) == y).mean())
    loss = float(-lp[np.arange(len(y)), y].mean())

    tag = f"d{a.dim}_s{a.seed}_sm{a.smooth}"
    np.savez_compressed(os.path.join(a.out, tag + '.npz'),
                        f=f.astype(np.float32), g=g.cpu().numpy().astype(np.float32),
                        logp=lp.astype(np.float32), y=y.astype(np.int64),
                        acc=acc, loss=loss, dim=a.dim, seed=a.seed,
                        smooth=a.smooth)
    print(json.dumps(dict(tag=tag, acc=acc, loss=loss)), flush=True)


if __name__ == '__main__':
    main()
