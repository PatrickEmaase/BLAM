"""
Corrected construction harness.

FIX 1  Labels assigned by ARGMAX (the power-diagram cell), not by construction
       index. Under non-uniform norms the winner is argmax_y lambda_y <f,u_y>,
       so index-based labelling was simply wrong.
FIX 2  Permute DIRECTIONS ONLY, keeping norms attached to the label index:
       g'(y_i) = lambda_i u_{sigma(i)}.  Then for x in cell j with
       R_j : u_j -> u_{sigma(j)},
          <f'(x), g'(y_j)> = lambda_j <R_j f, u_{sigma(j)}> = lambda_j <f, u_j>,
       so the TOP LOGIT IS PRESERVED EXACTLY and assumption (iv) holds.
       Reduces to the original construction when norms are uniform.
FIX 3  Irregular angles with a cyclic minimum separation; the old
       cumsum/sum*2pi put the last label on top of the first.
FIX 4  Spread expressed as a fraction of the cell inradius, with argmax
       re-verified after shrinking.
FIX 5  Full instrumentation; no bare excepts.
"""

import numpy as np

TOL = 1e-9


def rot_between(u, v, M):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    c = float(np.clip(u @ v, -1, 1))
    if c > 1 - 1e-12:
        return np.eye(M)
    w = v - c * u
    nw = np.linalg.norm(w)
    if nw < 1e-12:
        return -np.eye(M)
    w /= nw
    s = np.sqrt(max(0.0, 1 - c * c))
    R = np.eye(M) + (c - 1) * (np.outer(u, u) + np.outer(w, w))
    R += s * (np.outer(w, u) - np.outer(u, w))
    return R


def directions(M, k, rng, irregular=False, min_sep=0.4):
    """FIX 3: cyclic minimum separation, no wrap-around collision."""
    if M == 2:
        if irregular:
            for _ in range(200):
                th = np.sort(rng.uniform(0, 2 * np.pi, k))
                d = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
                if d.min() >= min_sep * 2 * np.pi / k:
                    break
        else:
            th = 2 * np.pi * np.arange(k) / k
        return np.stack([np.cos(th), np.sin(th)], 1)

    i = np.arange(k) + 0.5
    z = 1 - 2 * i / k
    r = np.sqrt(np.maximum(0, 1 - z * z))
    ph = np.pi * (1 + 5 ** 0.5) * i
    U = np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)
    if M > 3:
        U = np.concatenate([U, rng.normal(size=(k, M - 3)) * 0.35], 1)
    if irregular:
        U = U + rng.normal(size=U.shape) * 0.15
    return U / np.linalg.norm(U, axis=1, keepdims=True)


def make_pair(M, k, n_sample, rng, sigma=None, irregular=False,
              nonuniform=False, swap_dist=1, shrink=0.0, norm_spread=0.9):
    U = directions(M, k, rng, irregular)
    lam = 1 + norm_spread * rng.random(k) if nonuniform else np.ones(k)
    G = lam[:, None] * U                       # model 1 unembeddings

    if sigma is None:
        sigma = np.arange(k)
        for s in range(0, k - swap_dist, 2 * swap_dist):
            sigma[[s, s + swap_dist]] = sigma[[s + swap_dist, s]]
    sigma = np.asarray(sigma)

    Gp = lam[:, None] * U[sigma]               # FIX 2: directions only

    # FIX 1: sample on the sphere, label by argmax
    F = rng.normal(size=(n_sample, M))
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    lab = np.argmax(F @ G.T, 1)

    if shrink > 0:                             # FIX 4
        F2c = (1 - shrink) * F + shrink * U[lab]
        F2c /= np.linalg.norm(F2c, axis=1, keepdims=True)
        keep = np.argmax(F2c @ G.T, 1) == lab
        F, lab = F2c[keep], lab[keep]

    Fp = np.empty_like(F)
    for j in np.unique(lab):
        R = rot_between(U[j], U[sigma[j]], M)
        m = lab == j
        Fp[m] = F[m] @ R.T
    return F, G, Fp, Gp, lab, sigma, U, lam


def gaps_and_margin(F, G, Fp, Gp):
    t1 = F @ G.T
    t2 = Fp @ Gp.T
    top1 = np.argmax(t1, 1)
    top2 = np.argmax(t2, 1)
    ok = top1 == top2
    n = np.arange(len(F))
    a = t1[n, top1][:, None] - t1
    b = t2[n, top2][:, None] - t2
    off = np.ones_like(a, bool)
    off[n, top1] = False
    margin = np.where(off, a, np.inf).min(1)
    return a, b, off, margin, ok


def alpha_star_m(a, b, off, margin, ok, m):
    sel = ok & (margin >= m)
    if sel.sum() < 5:
        return np.nan, int(sel.sum())
    A = a[sel][off[sel]]; B = b[sel][off[sel]]
    msk = B > A + 1e-8
    if not msk.any():
        return np.inf, int(sel.sum())
    return float(np.min(B[msk] / (B[msk] - A[msk]))), int(sel.sum())


def _std(Z):
    Z = Z - Z.mean(0, keepdims=True)
    s = Z.std(0, keepdims=True); s[s < 1e-12] = 1.0
    return Z / s


def d_svd(Z, W):
    C = _std(Z).T @ _std(W) / (len(Z) - 1)
    return 1.0 - np.linalg.svd(C, compute_uv=False).mean()


def dfg(F, G, Fp, Gp, lab, M):
    present = [j for j in range(len(G)) if (lab == j).sum() > 0]
    if len(present) < M + 1:
        return np.nan
    L = (G - G[0])[1:M + 1].T
    Lp = (Gp - Gp[0])[1:M + 1].T
    d1 = d_svd(F @ L, Fp @ Lp)
    idx = [np.where(lab == j)[0][0] for j in present[:M + 1]]
    N = (F - F[idx[0]])[idx[1:]].T
    Np = (Fp - Fp[idx[0]])[idx[1:]].T
    return max(d1, d_svd(G @ N, Gp @ Np))


# --------------------------------------------------------- option 2 sweep ----
if __name__ == "__main__":
    MF = 0.08
    print("=" * 80)
    print("OPTION 2 RERUN with fixes 1-4   (margin floor %.2f)" % MF)
    print("=" * 80)
    print(f"{'irreg':>6} {'nonunif':>8} {'accept':>8} {'argmax':>8} "
          f"{'mean ok':>8} {'best a*_m':>10} {'d_fg there':>11}")
    for irreg in [False, True]:
        for nu in [False, True]:
            acc = rej = 0; oks = []; best = 0.0; bd = 0.0
            for trial in range(120):
                rng = np.random.default_rng(5000 + 17 * trial
                                            + 3 * int(irreg) + int(nu))
                k = int(rng.choice([8, 12, 18, 26]))
                sd = int(rng.choice([1, 2, 3]))
                sh = float(rng.choice([0.0, 0.25, 0.5]))
                F, G, Fp, Gp, lab, sg, U, lam = make_pair(
                    2, k, 1600, rng, irregular=irreg, nonuniform=nu,
                    swap_dist=sd, shrink=sh)
                a, b, off, margin, ok = gaps_and_margin(F, G, Fp, Gp)
                oks.append(ok.mean())
                if ok.mean() < 0.6:
                    rej += 1
                    continue
                A, ns = alpha_star_m(a, b, off, margin, ok, MF)
                d = dfg(F, G, Fp, Gp, lab, 2)
                if not np.isfinite(A) or ns < 30 or not np.isfinite(d):
                    rej += 1
                    continue
                acc += 1
                if d >= 0.10 and A > best:
                    best, bd = A, d
            print(f"{str(irreg):>6} {str(nu):>8} {acc:>8} {rej:>8} "
                  f"{np.mean(oks):>8.3f} {best:>10.4f} {bd:>11.4f}")
