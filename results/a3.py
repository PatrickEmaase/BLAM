"""
A3, options 1-4.

Option 4 (done properly): alpha*_m := min{ alpha*(x) : margin(x) >= m },
   margin(x) := min_{y != yhat} a_y   (logit-gap distance to the nearest
   decision boundary). Reporting alpha* conditioned on a margin floor turns the
   boundary degeneracy from a nuisance into a measured quantity.

Option 2: break the circular symmetry -- irregular angular placement and
   non-uniform unembedding norms, still M=2.

Option 3: M > 2, run only if option 2 looks promising.

GENERALISATION USED THROUGHOUT. The rate lemma needs only that both models
share the argmax at each input: p(y|x) ∝ exp(-rho a_y) and p'(y|x) ∝
exp(-rho b_y) after the common top logit cancels in the normaliser. The top
logits need NOT match. This frees the search considerably relative to the
original angle-preserving construction.
"""

import numpy as np

TOL = 1e-9


# --------------------------------------------------------------- geometry ---
def rot_between(u, v, M):
    """Rotation in span{u, v} taking unit u to unit v (identity elsewhere)."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    c = float(np.clip(u @ v, -1, 1))
    if c > 1 - 1e-12:
        return np.eye(M)
    w = v - c * u
    nw = np.linalg.norm(w)
    if nw < 1e-12:
        return -np.eye(M)
    w /= nw
    s = np.sqrt(max(0.0, 1 - c * c))
    R = np.eye(M)
    R += (c - 1) * (np.outer(u, u) + np.outer(w, w))
    R += s * (np.outer(w, u) - np.outer(u, w))
    return R


def make_pair(M, k, n_per, rng, spread=0.22, irregular=False,
              nonuniform=False, swap_dist=1):
    """Returns embeddings/unembeddings for the two models, same argmax by design."""
    if M == 2:
        if irregular:
            gaps = rng.uniform(0.4, 1.6, k)
            th = np.cumsum(gaps) / gaps.sum() * 2 * np.pi
        else:
            th = 2 * np.pi * np.arange(k) / k
        U = np.stack([np.cos(th), np.sin(th)], 1)
    else:
        i = np.arange(k) + 0.5
        z = 1 - 2 * i / k
        r = np.sqrt(np.maximum(0, 1 - z * z))
        ph = np.pi * (1 + 5 ** 0.5) * i
        U = np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)
        if M > 3:
            U = np.concatenate([U, rng.normal(size=(k, M - 3)) * 0.35], 1)
        if irregular:
            U = U + rng.normal(size=U.shape) * 0.18
        U /= np.linalg.norm(U, axis=1, keepdims=True)

    lam = 1 + 0.9 * rng.random(k) if nonuniform else np.ones(k)
    G = lam[:, None] * U

    sigma = np.arange(k)
    for s in range(0, k - swap_dist, 2 * swap_dist):
        sigma[[s, s + swap_dist]] = sigma[[s + swap_dist, s]]
    Gp = G[sigma]

    F1, F2, lab = [], [], []
    for jj in range(k):
        d = rng.normal(size=(n_per, M)) * spread
        v = U[jj] + d
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        R = rot_between(U[jj], U[sigma[jj]], M)
        F1.append(v); F2.append(v @ R.T); lab.append(np.full(n_per, jj))
    return np.vstack(F1), G, np.vstack(F2), Gp, np.concatenate(lab)


# ----------------------------------------------------------------- measures --
def gaps_and_margin(F1, G, F2, Gp):
    t1 = F1 @ G.T
    t2 = F2 @ Gp.T
    top1 = np.argmax(t1, 1)
    top2 = np.argmax(t2, 1)
    ok = top1 == top2                      # rate lemma requires shared argmax
    n = np.arange(len(F1))
    a = t1[n, top1][:, None] - t1
    b = t2[n, top2][:, None] - t2
    off = np.ones_like(a, bool)
    off[n, top1] = False
    margin = np.where(off, a, np.inf).min(1)
    return a, b, off, margin, ok


def alpha_star_m(a, b, off, margin, ok, m):
    sel = ok & (margin >= m)
    if not sel.any():
        return np.nan, 0
    A = a[sel][off[sel]]
    B = b[sel][off[sel]]
    msk = B > A + 1e-8
    if not msk.any():
        return np.inf, int(sel.sum())
    return float(np.min(B[msk] / (B[msk] - A[msk]))), int(sel.sum())


def _std(Z):
    Z = Z - Z.mean(0, keepdims=True)
    s = Z.std(0, keepdims=True)
    s[s < 1e-12] = 1.0
    return Z / s


def d_svd(Z, W):
    Zs, Ws = _std(Z), _std(W)
    C = Zs.T @ Ws / (len(Zs) - 1)
    return 1.0 - np.linalg.svd(C, compute_uv=False).mean()


def dfg(F1, G, F2, Gp, lab, M):
    L = (G - G[0])[1:M + 1].T
    Lp = (Gp - Gp[0])[1:M + 1].T
    d1 = d_svd(F1 @ L, F2 @ Lp)
    idx = [np.where(lab == j)[0][0] for j in range(M + 1)]
    N = (F1 - F1[idx[0]])[idx[1:]].T
    Np = (F2 - F2[idx[0]])[idx[1:]].T
    return max(d1, d_svd(G @ N, Gp @ Np))


# ------------------------------------------------- option 4: margin sweep ----
print("=" * 78)
print("OPTION 4   alpha*_m = min{alpha*(x) : margin(x) >= m}")
print("=" * 78)
rng = np.random.default_rng(0)
F1, G, F2, Gp, lab = make_pair(2, 12, 60, rng, spread=0.22)
a, b, off, margin, ok = gaps_and_margin(F1, G, F2, Gp)
print(f"margin range [{margin.min():.4f}, {margin[np.isfinite(margin)].max():.4f}]"
      f"   shared-argmax fraction {ok.mean():.3f}")
print(f"{'m':>8} {'alpha*_m':>10} {'#inputs kept':>13}")
for m in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    A, nsel = alpha_star_m(a, b, off, margin, ok, m)
    print(f"{m:>8.2f} {A:>10.4f} {nsel:>13d}")

# ------------------------------- option 2 / 3: symmetry-breaking search ------
print("\n" + "=" * 78)
print("OPTIONS 2 & 3   does breaking symmetry raise alpha* at a fixed margin?")
print("=" * 78)
print(f"{'M':>3} {'irreg':>6} {'nonunif':>8} {'best a*_0':>10} "
      f"{'best a*_0.1':>12} {'d_fg at best':>13}")

MFLOOR = 0.10
summary = {}
for M in [2, 3, 4, 5]:
    for irreg in [False, True]:
        for nu in [False, True]:
            if M == 2 and not irreg and not nu:
                tag = "baseline"
            best0, bestm, bestd = 0.0, 0.0, 0.0
            for trial in range(120):
                rng = np.random.default_rng(1000 * M + 10 * int(irreg)
                                            + int(nu) + 7 * trial)
                k = int(rng.choice([8, 12, 18, 26]))
                sd = int(rng.choice([1, 2, 3]))
                sp = float(rng.choice([0.12, 0.2, 0.3]))
                try:
                    F1, G, F2, Gp, lab = make_pair(M, k, 24, rng, spread=sp,
                                                   irregular=irreg,
                                                   nonuniform=nu, swap_dist=sd)
                    a, b, off, margin, ok = gaps_and_margin(F1, G, F2, Gp)
                    if ok.mean() < 0.9:
                        continue
                    A0, _ = alpha_star_m(a, b, off, margin, ok, 0.0)
                    Am, ns = alpha_star_m(a, b, off, margin, ok, MFLOOR)
                    if ns < 20:
                        continue
                    d = dfg(F1, G, F2, Gp, lab, M)
                    if d < 0.02:          # require genuine dissimilarity
                        continue
                    if np.isfinite(A0):
                        best0 = max(best0, A0)
                    if np.isfinite(Am) and Am > bestm:
                        bestm, bestd = Am, d
                except Exception:
                    continue
            summary[(M, irreg, nu)] = (best0, bestm, bestd)
            print(f"{M:>3} {str(irreg):>6} {str(nu):>8} {best0:>10.4f} "
                  f"{bestm:>12.4f} {bestd:>13.4f}")

b2 = max(v[1] for (M, _, _), v in summary.items() if M == 2)
b3 = max(v[1] for (M, _, _), v in summary.items() if M >= 3)
print(f"\nbest alpha*_m  M=2: {b2:.4f}    M>=3: {b3:.4f}")
print(f"(requiring d_fg >= 0.02 and margin floor m = {MFLOOR})")
