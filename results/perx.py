"""
Gate 1 decisive check.

The rate lemma is a PER-INPUT statement. Verify it at a single x (exact), then
show what averaging over x does to it -- which turns out to be a separate
phenomenon worth its own result.
"""

import numpy as np
from scipy.special import logsumexp

TOL = 1e-9


# ---------------------------------------------------------------- one input --
def one_x(k, phi, j, sigma):
    """Cosine gap vectors (a, b) at a single input in cluster j with offset phi."""
    th = 2 * np.pi * np.arange(k) / k
    s1 = np.cos(th[j] + phi - th)                  # model 1
    s2 = np.cos(th[sigma[j]] + phi - th[sigma])    # model 2
    a = s1[j] - s1
    b = s2[j] - s2
    off = np.arange(k) != j
    return a[off], b[off], a, b


def D_alpha_exact(a_full, b_full, rho, alpha):
    lp = -rho * a_full
    lp -= logsumexp(lp)
    lq = -rho * b_full
    lq -= logsumexp(lq)
    if np.isinf(alpha):
        return np.max(lp - lq)
    if abs(alpha - 1) < TOL:
        return np.sum(np.exp(lp) * (lp - lq))
    return logsumexp(alpha * lp + (1 - alpha) * lq) / (alpha - 1)


def rate_pred(a, b, alpha):
    keep = np.abs(a - b) > 1e-8
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return np.inf
    if abs(alpha - 1) < TOL:
        return a.min()
    E = b + alpha * (a - b)
    vals = np.concatenate([E, a, b])
    coef = np.concatenate([np.ones_like(E), -alpha * np.ones_like(a),
                           (alpha - 1) * np.ones_like(b)])
    o = np.argsort(vals)
    vals, coef = vals[o], coef[o]
    i = 0
    while i < len(vals):
        j, tot = i, 0.0
        while j < len(vals) and vals[j] - vals[i] < 1e-7:
            tot += coef[j]
            j += 1
        if abs(tot) > 1e-8 and vals[i] > TOL:
            return vals[i]
        i = j
    return np.inf


def slope_pred(a, b, alpha):
    keep = np.abs(a - b) > 1e-8
    a, b = a[keep], b[keep]
    if np.isinf(alpha):
        return (b - a).max()
    E = b + alpha * (a - b)
    return max(0.0, -E.min()) / (alpha - 1)


def astar(a, b):
    m = b > a + 1e-8
    return np.min(b[m] / (b[m] - a[m])) if m.any() else np.inf


# --------------------------------------------------------------------- main --
def test_single_x(k=7, perm_seed=1, phi=0.25 * np.pi / 7, j=0):
    rng = np.random.default_rng(perm_seed)
    sigma = rng.permutation(k)
    a, b, af, bf = one_x(k, phi, j, sigma)
    ast = astar(a, b)

    print("=" * 72)
    print(f"PER-INPUT TEST   k={k}  sigma={sigma.tolist()}  phi={phi:.4f}")
    print(f"predicted alpha* = {ast:.6f}")
    print("=" * 72)

    rhos = np.array([40, 60, 80, 100, 120, 140], float)   # deep asymptotic
    alphas = [0.3, 0.5, 0.9, 1.0, 1.02, 1.05, 1.1, 1.3, 1.6,
              2.0, 4.0, 10.0, np.inf]

    print(f"{'alpha':>7} {'branch':>7} {'pred':>11} {'fit':>11} {'rel.err':>9}")
    worst_exp = worst_lin = 0.0
    for al in alphas:
        y = np.array([D_alpha_exact(af, bf, r, al) for r in rhos])
        if y[-1] > y[0] + 1e-12:
            pred = slope_pred(a, b, al)
            fit = np.polyfit(rhos, y, 1)[0]
            br = "linear"
            worst_lin = max(worst_lin, abs(fit - pred) / max(pred, 1e-12))
        else:
            pred = rate_pred(a, b, al)
            adj = y / rhos if abs(al - 1) < TOL else y
            fit = -np.polyfit(rhos, np.log(np.abs(adj)), 1)[0]
            br = "exp"
            worst_exp = max(worst_exp, abs(fit - pred) / max(pred, 1e-12))
        nm = "inf" if np.isinf(al) else f"{al:g}"
        print(f"{nm:>7} {br:>7} {pred:>11.6f} {fit:>11.6f} "
              f"{abs(fit-pred)/max(abs(pred),1e-12):>9.3%}")

    print(f"\nworst rel. error   exp branch: {worst_exp:.3%}   "
          f"linear branch: {worst_lin:.3%}")

    # bisect the empirical crossover
    lo, hi = 1.0, 20.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if D_alpha_exact(af, bf, 140, mid) > D_alpha_exact(af, bf, 40, mid):
            hi = mid
        else:
            lo = mid
    print(f"bisected crossover = {0.5*(lo+hi):.6f}   predicted = {ast:.6f}   "
          f"rel.err = {abs(0.5*(lo+hi)-ast)/ast:.4%}")
    return ast


def test_alphastar_spread(k=7, n_perm=40):
    """Does alpha* vary with geometry, and does it trade off against d_fg?"""
    print("\n" + "=" * 72)
    print("ALPHA* VS REPRESENTATIONAL SEPARATION")
    print("=" * 72)
    rows = []
    for ps in range(1, n_perm + 1):
        rng = np.random.default_rng(ps)
        sigma = rng.permutation(k)
        if np.all(sigma == np.arange(k)):
            continue
        for phi in np.linspace(-0.7, 0.7, 9) * np.pi / k:
            a, b, af, bf = one_x(k, phi, 0, sigma)
            ast = astar(a, b)
            if np.isfinite(ast):
                # sup log-ratio growth rate = representational "signal"
                sig = (b - a).max()
                rows.append((ast, sig))
    A = np.array(rows)
    print(f"{len(A)} (permutation, offset) configurations")
    print(f"alpha* : min={A[:,0].min():.3f}  median={np.median(A[:,0]):.3f}  "
          f"max={A[:,0].max():.3f}")
    r = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
    print(f"corr(alpha*, max_y(b-a)) = {r:+.4f}")
    hi = A[A[:, 0] > np.percentile(A[:, 0], 80)]
    lo = A[A[:, 0] < np.percentile(A[:, 0], 20)]
    print(f"top-20% alpha*: mean signal = {hi[:,1].mean():.4f}")
    print(f"bot-20% alpha*: mean signal = {lo[:,1].mean():.4f}")
    return A


def test_averaging(k=7, perm_seed=1, n=4000):
    """Averaging over a continuum of x: exponential -> power law."""
    print("\n" + "=" * 72)
    print("EFFECT OF AVERAGING OVER x")
    print("=" * 72)
    rng = np.random.default_rng(perm_seed)
    sigma = rng.permutation(k)
    rhos = np.array([20, 40, 60, 80, 120, 160], float)

    for name, hw in [("margin 0.5*pi/k (bounded away)", 0.5),
                     ("margin 0.95*pi/k (touches boundary)", 0.95)]:
        phis = np.linspace(-hw, hw, n) * np.pi / k
        kl = []
        for r in rhos:
            v = 0.0
            for p in phis:
                _, _, af, bf = one_x(k, p, 0, sigma)
                v += D_alpha_exact(af, bf, r, 1.0)
            kl.append(v / len(phis))
        kl = np.array(kl)
        exp_fit = -np.polyfit(rhos, np.log(kl), 1)[0]
        pow_fit = -np.polyfit(np.log(rhos), np.log(kl), 1)[0]
        r2e = np.corrcoef(rhos, np.log(kl))[0, 1] ** 2
        r2p = np.corrcoef(np.log(rhos), np.log(kl))[0, 1] ** 2
        print(f"\n{name}")
        print(f"  exponential fit: rate={exp_fit:.5f}  R^2={r2e:.5f}")
        print(f"  power-law   fit: exp ={pow_fit:.5f}  R^2={r2p:.5f}")
        print(f"  -> {'POWER LAW' if r2p > r2e else 'EXPONENTIAL'} dominates")


if __name__ == "__main__":
    test_single_x()
    test_alphastar_spread()
    test_averaging()
