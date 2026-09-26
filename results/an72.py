"""Section 7.2 analysis. Real results on the trained synthetic models."""

import pickle
import numpy as np
from scipy.stats import spearmanr
from scipy.special import logsumexp
from train72 import make_data, logprob, embed

M = 2
models = pickle.load(open("models.pkl", "rb"))


def _std(Z):
    Z = Z - Z.mean(0, keepdims=True)
    s = Z.std(0, keepdims=True); s[s < 1e-12] = 1.0
    return Z / s


def d_svd(Z, W):
    C = _std(Z).T @ _std(W) / (len(Z) - 1)
    return 1.0 - np.linalg.svd(C, compute_uv=False).mean()


def mcca(Z, W):
    Zs, Ws = _std(Z), _std(W)
    n = len(Zs)
    A = np.linalg.inv(np.linalg.cholesky(Zs.T @ Zs / (n - 1) + 1e-9 * np.eye(M)))
    B = np.linalg.inv(np.linalg.cholesky(Ws.T @ Ws / (n - 1) + 1e-9 * np.eye(M)))
    return np.linalg.svd(A @ (Zs.T @ Ws / (n - 1)) @ B.T, compute_uv=False).mean()


def pair_stats(P1, P2, X, y, c):
    f1, _ = embed(P1, X); f2, _ = embed(P2, X)
    lp, lq = logprob(P1, X), logprob(P2, X)

    kl = float(np.mean(np.sum(np.exp(lp) * (lp - lq), 1)))
    tv = float(np.mean(.5 * np.abs(np.exp(lp) - np.exp(lq)).sum(1)))
    ren = {}
    for al in [0.5, 1.5, 2.0, 4.0]:
        ren[al] = float(np.mean(logsumexp(al * lp + (1 - al) * lq, 1) / (al - 1)))
    ren[np.inf] = float(np.mean(np.max(lp - lq, 1)))

    G1, G2 = P1['G'], P2['G']
    L = (G1 - G1[0])[1:M + 1].T
    Lp = (G2 - G2[0])[1:M + 1].T
    idx = [np.where(y == j)[0][0] for j in range(M + 1)]
    N = (f1 - f1[idx[0]])[idx[1:]].T
    Np = (f2 - f2[idx[0]])[idx[1:]].T
    d = max(d_svd(f1 @ L, f2 @ Lp), d_svd(G1 @ N, G2 @ Np))

    ps = np.sort(np.exp(lp), 1)[:, ::-1]
    dm = float(ps[:, :M + 1].min())
    z1 = lp[:, 1:M + 1] - lp[:, [0]]
    z2 = lq[:, 1:M + 1] - lq[:, [0]]
    rho2 = float((z1 - z2).std(0).max())
    psi = float(min(z1.std(0).min(), z2.std(0).min()))
    Zc = _std(z1)
    lm = float(np.linalg.eigvalsh(Zc.T @ Zc / (len(z1) - 1)).max())
    bound = 2 * np.sqrt(lm) * rho2 / psi
    return dict(kl=kl, tv=tv, **{f"a{a}": v for a, v in ren.items()},
                dfg=d, mcca=float(mcca(f1, f2)), dm=dm, bound=bound)


# ----------------------------------------------------- 7.2a width sweep -----
print("=" * 78)
print("7.2a  WIDTH SWEEP, and the matched-loss control")
print("=" * 78)
print(f"{'c':>2} {'w':>4} {'mean loss':>10} {'mean d_KL':>10} {'sd d_KL':>9} "
      f"{'mean d_fg':>10} {'sd d_fg':>9}")
rows = []
for c in [4, 6]:
    X, y = make_data(c, seed=0)
    for w in [16, 32, 64, 128, 256]:
        st, ls = [], []
        for i in range(6):
            for j in range(i + 1, 6):
                P1, a1, l1 = models[('A', c, w, 0.0, i)]
                P2, a2, l2 = models[('A', c, w, 0.0, j)]
                s = pair_stats(P1, P2, X, y, c)
                s['loss'] = 0.5 * (l1 + l2); s['c'] = c; s['w'] = w
                st.append(s); ls.append(s['loss'])
        kl = np.array([s['kl'] for s in st]); dd = np.array([s['dfg'] for s in st])
        rows += st
        print(f"{c:>2} {w:>4} {np.mean(ls):>10.4f} {kl.mean():>10.4f} "
              f"{kl.std():>9.4f} {dd.mean():>10.4f} {dd.std():>9.4f}")

import numpy.linalg as la
print("\nregression of d_fg on log(width) with mean training loss as covariate")
for c in [4, 6]:
    R = [s for s in rows if s['c'] == c]
    Xd = np.column_stack([np.ones(len(R)), np.log([s['w'] for s in R]),
                          [s['loss'] for s in R]])
    yv = np.array([s['dfg'] for s in R])
    beta, *_ = la.lstsq(Xd, yv, rcond=None)
    Xn = np.column_stack([np.ones(len(R)), np.log([s['w'] for s in R])])
    bn, *_ = la.lstsq(Xn, yv, rcond=None)
    print(f"  c={c}: naive slope(log w) = {bn[1]:+.4f}   "
          f"loss-adjusted = {beta[1]:+.4f}   loss coef = {beta[2]:+.4f}")

# ----------------------------------------------------- 7.2b control curves --
print("\n" + "=" * 78)
print("7.2b  CONTROL CURVES  h(eps) = sup{ d_fg : D <= eps }")
print("=" * 78)
keys = [('kl', 'KL'), ('tv', 'TV'), ('a0.5', 'Renyi 0.5'), ('a1.5', 'Renyi 1.5'),
        ('a2.0', 'Renyi 2'), ('a4.0', 'Renyi 4'), ('ainf', 'Renyi inf')]
qs = [0.05, 0.1, 0.25, 0.5]
print(f"{'divergence':>11} " + " ".join(f"{'q=' + str(q):>9}" for q in qs)
      + f" {'h(q05)/h(q50)':>14}")
for k, nm in keys:
    v = np.array([s[k] for s in rows]); d = np.array([s['dfg'] for s in rows])
    hs = []
    for q in qs:
        th = np.quantile(v, q)
        sel = v <= th
        hs.append(d[sel].max() if sel.any() else np.nan)
    print(f"{nm:>11} " + " ".join(f"{h:>9.4f}" for h in hs)
          + f" {hs[0]/hs[-1]:>14.3f}")
print("\nratio near 1 => the modulus does not fall as eps -> 0 (no control)")

# ------------------------------------------------- 7.2c smoothing arm -------
print("\n" + "=" * 78)
print("7.2c  LABEL SMOOTHING: does raising delta_m restore KL?")
print("=" * 78)
X, y = make_data(6, seed=0)
print(f"{'smooth':>7} {'delta_m':>11} {'mean d_KL':>10} {'mean d_fg':>10} "
      f"{'spearman(KL,d_fg)':>18} {'p':>8}")
for sm in [0.0, 0.05, 0.1, 0.2]:
    st = []
    for i in range(6):
        for j in range(i + 1, 6):
            P1 = models[('B', 6, 64, sm, i)][0]
            P2 = models[('B', 6, 64, sm, j)][0]
            st.append(pair_stats(P1, P2, X, y, 6))
    kl = np.array([s['kl'] for s in st]); dd = np.array([s['dfg'] for s in st])
    dm = np.median([s['dm'] for s in st])
    r, p = spearmanr(kl, dd)
    print(f"{sm:>7} {dm:>11.3e} {kl.mean():>10.4f} {dd.mean():>10.4f} "
          f"{r:>18.3f} {p:>8.3f}")

# ------------------------------------------------- 7.2d non-vacuity --------
print("\n" + "=" * 78)
print("7.2d  NON-VACUITY of the sharpened bound (Thm 12)")
print("=" * 78)
b = np.array([s['bound'] for s in rows]); d = np.array([s['dfg'] for s in rows])
print(f"pairs: {len(b)}   bound < 1 (non-vacuous): {(b < 1).mean():.1%}")
print(f"violations d_fg > bound: {(d > b + 1e-9).sum()}")
print(f"median bound = {np.median(b):.3f}   median d_fg = {np.median(d):.3f}")
for c in [4, 6]:
    sel = np.array([s['c'] == c for s in rows])
    print(f"  c={c}: non-vacuous {(b[sel] < 1).mean():.1%}, "
          f"median bound {np.median(b[sel]):.3f}")
np.save("rows72.npy", np.array(rows, dtype=object), allow_pickle=True)
