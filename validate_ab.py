"""Two validations from existing .npz files.

A. Positive control for the dichotomy (Thm 11). Applying a known invertible A
   (f -> A f, g_0 -> A^{-T} g_0) gives a pair that IS ~_L-equivalent, so the
   theory demands alpha*_sym = inf, d_fg < 5/n, and D_alpha = 0 at every order.
B. Graded control (Thm 12) and the ceiling (Cor 13) on trained pairs, having
   been validated only on constructed geometries so far.

    python validate_ab.py runs/cifar/d2_s*_sm0.0.npz
"""
import sys
import glob
import itertools

import numpy as np
from scipy.special import logsumexp

EPS = 1e-12


def _std(Z):
    Z = Z - Z.mean(0)
    s = Z.std(0)
    s[s < EPS] = 1.0
    return Z / s


def d_svd(Z, W):
    C = _std(Z).T @ _std(W) / (len(Z) - 1)
    return float(1 - np.linalg.svd(C, compute_uv=False).mean())


def d_fg(f1, g1, f2, g2, y, M):
    ys = list(range(M + 1))
    L = (g1 - g1[0])[ys[1:]].T
    Lp = (g2 - g2[0])[ys[1:]].T
    idx = [int(np.where(y == j)[0][0]) for j in ys]
    N = (f1 - f1[idx[0]])[idx[1:]].T
    Np = (f2 - f2[idx[0]])[idx[1:]].T
    return max(d_svd(f1 @ L, f2 @ Lp), d_svd(g1 @ N, g2 @ Np))


def logp(f, g):
    lg = f @ g.T
    return lg - logsumexp(lg, 1, keepdims=True)


def d_renyi(lp, lq, a):
    if np.isinf(a):
        return float(np.mean(np.max(lp - lq, 1)))
    if abs(a - 1) < EPS:
        p = np.exp(lp)
        m = p > 0
        d = np.zeros_like(lp)
        d[m] = lp[m] - lq[m]
        return float(np.mean(np.sum(p * d, 1)))
    return float(np.mean(logsumexp(a * lp + (1 - a) * lq, 1) / (a - 1)))


def alpha_star_sym(lp, lq, m=0.0):
    t1, t2 = lp.argmax(1), lq.argmax(1)
    ok = t1 == t2
    n = np.arange(len(lp))
    a = lp[n, t1][:, None] - lp
    b = lq[n, t2][:, None] - lq
    off = np.ones_like(a, bool)
    off[n, t1] = False
    mar = np.where(off, a, np.inf).min(1)
    sel = ok & (mar >= m)
    if sel.sum() < 5:
        return np.nan, 0.0
    A = a[sel][off[sel]]
    B = b[sel][off[sel]]

    def one(P, Q):
        msk = Q > P + 1e-8
        return float(np.min(Q[msk] / (Q - P)[msk])) if msk.any() else np.inf

    return min(one(A, B), one(B, A)), float(ok.mean())


def geom(lp, lq, M):
    z1 = lp[:, 1:M + 1] - lp[:, [0]]
    z2 = lq[:, 1:M + 1] - lq[:, [0]]
    psi = float(min(z1.std(0).min(), z2.std(0).min()))
    top = lp.argmax(1)
    n = np.arange(len(lp))
    a = lp[n, top][:, None] - lp
    off = np.ones_like(a, bool)
    off[n, top] = False
    Delta = float(np.where(off, a, 0).max())
    return Delta, psi


def main():
    paths = sys.argv[1:] or sorted(glob.glob('runs/cifar/d2_s*_sm0.0.npz'))
    if not paths:
        print("no .npz files found")
        return
    Z = [np.load(p) for p in paths]
    M = int(Z[0]['dim'])
    N = Z[0]['logp'].shape[0]
    print(f"{len(Z)} models, M={M}, N={N} inputs, tolerance 5/n = {5 / N:.3e}\n")

    print("=" * 72)
    print("A. POSITIVE CONTROL: apply a known invertible A; the pair IS ~_L")
    print("=" * 72)
    rng = np.random.default_rng(0)
    print(f"{'seed':>5} {'cond(A)':>9} {'d_fg':>11} {'< 5/n':>7} "
          f"{'alpha*_sym':>11} {'max|D_a|':>10}")
    bad_d = bad_a = bad_D = 0
    for zz in Z[:8]:
        f = zz['f'].astype(float)
        g = zz['g'].astype(float)
        y = zz['y']
        A = rng.normal(size=(M, M))
        while abs(np.linalg.det(A)) < 0.1:
            A = rng.normal(size=(M, M))
        f2 = f @ A.T
        g2 = g @ np.linalg.inv(A)
        d = abs(d_fg(f, g, f2, g2, y, M))
        lp, lq = logp(f, g), logp(f2, g2)
        ast, _ = alpha_star_sym(lp, lq)
        Ds = [abs(d_renyi(lp, lq, a)) for a in (0.5, 1.0, 2.0, np.inf)]
        okd = d < 5 / N
        oka = np.isinf(ast)
        okD = max(Ds) < 1e-8
        bad_d += not okd
        bad_a += not oka
        bad_D += not okD
        shown = 'inf' if oka else f'{ast:.3f}'
        print(f"{int(zz['seed']):>5} {np.linalg.cond(A):>9.2f} {d:>11.3e} "
              f"{'yes' if okd else 'NO':>7} {shown:>11} {max(Ds):>10.2e}")
    print(f"\n  d_fg < 5/n failures        : {bad_d}/8   (must be 0)")
    print(f"  alpha*_sym != inf failures : {bad_a}/8   (must be 0)")
    print(f"  D_alpha != 0 failures      : {bad_D}/8   (must be 0)")

    print("\n" + "=" * 72)
    print("B. GRADED CONTROL (Thm 12) and THE CEILING (Cor 13) on trained pairs")
    print("=" * 72)
    v12 = v13 = tot = 0
    slack, caps = [], []
    for i, j in itertools.combinations(range(len(Z)), 2):
        Ai, Bj = Z[i], Z[j]
        f1 = Ai['f'].astype(float)
        g1 = Ai['g'].astype(float)
        f2 = Bj['f'].astype(float)
        g2 = Bj['g'].astype(float)
        lp, lq = logp(f1, g1), logp(f2, g2)
        ast, _ = alpha_star_sym(lp, lq)
        if not np.isfinite(ast):
            continue
        d = d_fg(f1, g1, f2, g2, Ai['y'], M)
        Delta, psi = geom(lp, lq, M)
        tot += 1
        bnd12 = 4 * np.sqrt(M) * Delta / (psi * ast)
        v12 += d > bnd12 + 1e-9
        slack.append(bnd12 / max(d, 1e-12))
        if d > 1e-6:
            cap = 4 * np.sqrt(M) * Delta / (psi * d)
            caps.append(cap / ast)
            v13 += cap < ast - 1e-9
    print(f"  pairs tested                       : {tot}")
    print(f"  Thm 12 violations                  : {v12}   (must be 0)")
    print(f"  Cor 13 violations                  : {v13}   (must be 0)")
    print(f"  Thm 12 slack  bound/d_fg           : median "
          f"{np.median(slack):.1f}x, min {np.min(slack):.2f}x")
    print(f"  Cor 13 cap / measured alpha*_sym   : median "
          f"{np.median(caps):.1f}x, min {np.min(caps):.2f}x")


if __name__ == '__main__':
    main()
