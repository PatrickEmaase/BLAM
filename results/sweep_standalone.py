import sys, glob
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

def _std(Z):
    Z = Z - Z.mean(0, keepdims=True); s = Z.std(0, keepdims=True)
    s[s < 1e-12] = 1.0; return Z / s

def d_svd(Z, W):
    C = _std(Z).T @ _std(W) / (len(Z) - 1)
    return float(1 - np.linalg.svd(C, compute_uv=False).mean())

def d_kl(lp, lq):
    p = np.exp(lp); m = p > 0; d = np.zeros_like(lp)
    d[m] = lp[m] - lq[m]; return float(np.mean(np.sum(p * d, 1)))

def d_fg(f1, g1, f2, g2, y, M):
    ys = list(range(M + 1)); piv, rest = ys[0], ys[1:M + 1]
    L = (g1 - g1[piv])[rest].T; Lp = (g2 - g2[piv])[rest].T
    idx = [int(np.where(y == j)[0][0]) for j in ys]
    N = (f1 - f1[idx[0]])[idx[1:]].T; Np = (f2 - f2[idx[0]])[idx[1:]].T
    return max(d_svd(f1 @ L, f2 @ Lp), d_svd(g1 @ N, g2 @ Np))

paths = sys.argv[1:] or sorted(glob.glob('runs/cifar/*sm0.0.npz'))
R = []
for p in paths:
    z = np.load(p)
    R.append(dict(f=z['f'].astype(float), g=z['g'].astype(float),
                  lp=z['logp'].astype(float), y=z['y'], dim=int(z['dim']),
                  acc=float(z['acc'])))
M = R[0]['dim']
print(f"{len(R)} models, M={M}, mean acc {np.mean([m['acc'] for m in R]):.4f}\n")

A = R[0]
z1 = A['lp'][:, 1:M+1] - A['lp'][:, [0]]
psi = float(z1.std(0).min())
top = A['lp'].argmax(1); n = np.arange(len(A['lp']))
a = A['lp'][n, top][:, None] - A['lp']
off = np.ones_like(a, bool); off[n, top] = False
Delta = float(np.where(off, a, 0).max())
Zc = _std(z1); lam = float(np.linalg.eigvalsh(Zc.T @ Zc / (len(z1)-1)).max())
print("=== CEILINGS (item 5) ===")
print(f"  psi_min {psi:.4f}   Delta {Delta:.4f}   Delta/psi {Delta/psi:.3f}"
      f"   lambda_max {lam:.4f} (<= M={M})")
print(f"  Prop 13: bound<1 forces d_fg < {1/(2*lam):.4f}")
print(f"  Cor 19 : needs alpha > {4*np.sqrt(M)*Delta/psi:.1f}  vs ceiling ~1.35"
      f"  -> {4*np.sqrt(M)*Delta/psi/1.35:.0f}x out of reach")

if len(R) < 2:
    print("\nneed >= 2 models for the sweep"); sys.exit()
pairs = [(R[i], R[j]) for i in range(len(R)) for j in range(i+1, len(R))]
dref = [d_fg(x['f'], x['g'], y['f'], y['g'], x['y'], M) for x, y in pairs]
print(f"\n=== TEMPERATURE SWEEP (item 4), {len(pairs)} pairs ===")
print(f"  {'T':>5} {'delta_3':>11} {'mean KL':>9} {'mean d_fg':>10} {'spearman':>9} {'p':>8}")
for T in [1.0, 2.0, 4.0, 8.0, 16.0]:
    kl, dm = [], []
    for x, y in pairs:
        la = x['f'] @ x['g'].T / T; la -= logsumexp(la, 1, keepdims=True)
        lb = y['f'] @ y['g'].T / T; lb -= logsumexp(lb, 1, keepdims=True)
        kl.append(d_kl(la, lb))
        ps = np.sort(np.exp(la), 1)[:, ::-1]; dm.append(ps[:, :M+1].min())
    rho, pv = spearmanr(kl, dref)
    print(f"  {T:>5.1f} {np.median(dm):>11.3e} {np.mean(kl):>9.4f} "
          f"{np.mean(dref):>10.4f} {rho:>9.3f} {pv:>8.4f}")
hot = [d_fg(x['f']/8, x['g'], y['f']/8, y['g'], x['y'], M) for x, y in pairs]
print(f"\n  d_fg invariance check: max change {max(abs(u-v) for u,v in zip(dref,hot)):.2e}")
