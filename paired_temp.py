import sys, glob, itertools
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

def _s(Z):
    Z = Z - Z.mean(0); s = Z.std(0); s[s < 1e-12] = 1; return Z / s
def dsvd(Z, W):
    C = _s(Z).T @ _s(W) / (len(Z) - 1)
    return float(1 - np.linalg.svd(C, compute_uv=False).mean())
def dfg(f1, g1, f2, g2, y, M):
    ys = list(range(M + 1)); L = (g1 - g1[0])[ys[1:]].T; Lp = (g2 - g2[0])[ys[1:]].T
    idx = [int(np.where(y == j)[0][0]) for j in ys]
    N = (f1 - f1[idx[0]])[idx[1:]].T; Np = (f2 - f2[idx[0]])[idx[1:]].T
    return max(dsvd(f1 @ L, f2 @ Lp), dsvd(g1 @ N, g2 @ Np))
def dkl(lp, lq):
    p = np.exp(lp); m = p > 0; d = np.zeros_like(lp); d[m] = lp[m] - lq[m]
    return float(np.mean(np.sum(p * d, 1)))

paths = sys.argv[1:] or sorted(glob.glob('runs/cifar/d2_s*_sm0.0.npz'))
Z = [np.load(p) for p in paths]
M = int(Z[0]['dim']); n = len(Z)
print(f"{n} models, M={M}")

TS = [1.0, 2.0, 4.0, 8.0, 16.0]
pairs, D, KL = [], [], {T: [] for T in TS}
for i, j in itertools.combinations(range(n), 2):
    A, B = Z[i], Z[j]
    pairs.append((i, j))
    D.append(dfg(A['f'].astype(float), A['g'].astype(float),
                 B['f'].astype(float), B['g'].astype(float), A['y'], M))
    for T in TS:
        la = A['f'].astype(float) @ A['g'].astype(float).T / T
        la -= logsumexp(la, 1, keepdims=True)
        lb = B['f'].astype(float) @ B['g'].astype(float).T / T
        lb -= logsumexp(lb, 1, keepdims=True)
        KL[T].append(dkl(la, lb))
D = np.array(D); KL = {T: np.array(v) for T, v in KL.items()}

print(f"\n{'T':>5} {'rho':>8}   marginal 95% CI")
rng = np.random.default_rng(0)
B = 4000
boots = {T: [] for T in TS}
diffs = []
for _ in range(B):
    take = set(rng.choice(n, n, replace=True).tolist())
    sel = [k for k, (i, j) in enumerate(pairs) if i in take and j in take]
    if len(sel) < 5:
        continue
    rs = {}
    for T in TS:
        r = spearmanr(KL[T][sel], D[sel])[0]
        if np.isfinite(r):
            boots[T].append(r); rs[T] = r
    if 1.0 in rs and 16.0 in rs:
        diffs.append(rs[16.0] - rs[1.0])
for T in TS:
    lo, hi = np.percentile(boots[T], [2.5, 97.5])
    print(f"{T:>5.0f} {spearmanr(KL[T], D)[0]:>8.3f}   [{lo:>6.3f}, {hi:>6.3f}]")

d = np.array(diffs)
lo, hi = np.percentile(d, [2.5, 97.5])
obs = spearmanr(KL[16.0], D)[0] - spearmanr(KL[1.0], D)[0]
print(f"\nPAIRED: rho(T=16) - rho(T=1)")
print(f"  observed {obs:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  P(increase) = {np.mean(d > 0):.4f}   n_boot = {len(d)}")
print(f"  {'EXCLUDES zero -> the rise is significant' if lo > 0 else 'includes zero -> not significant'}")
