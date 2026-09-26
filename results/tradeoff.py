"""
The decisive question for the paper's architecture.

We know: for a given construction, alpha <= alpha* is defeated.
We do NOT know: whether alpha* can be driven arbitrarily large while the
representations stay far apart.

  - If YES: no finite alpha controls on the unrestricted class Theta, and
    alpha*_Theta = infinity. The negative result is sweeping; the positive
    result MUST be conditional (on Theta_delta).
  - If NO (alpha* large forces d_fg -> 0): the trade-off is real and can be
    turned into a genuine finite-order control modulus h(eps) ~ 1/alpha.

Knob: gentleness of the permutation. A transposition of two labels that are
close on the circle perturbs the margin geometry only slightly, which should
push a_y toward b_y and hence alpha* = min b/(b-a) upward.
"""

import numpy as np
from scipy.special import logsumexp
from perx import astar

TOL = 1e-9


def build_pair(k, swap_i, swap_j, n_per=60, rho=1.0, margin=0.6, seed=0):
    """Model 1 canonical; model 2 transposes labels swap_i, swap_j."""
    th = 2 * np.pi * np.arange(k) / k
    sigma = np.arange(k)
    sigma[[swap_i, swap_j]] = sigma[[swap_j, swap_i]]

    phi = np.linspace(-margin, margin, n_per) * np.pi / k
    lab = np.repeat(np.arange(k), n_per)
    ang1 = (th[:, None] + phi[None, :]).ravel()
    ang2 = (th[sigma][:, None] + phi[None, :]).ravel()

    f1 = np.stack([np.cos(ang1), np.sin(ang1)], 1)
    f2 = np.stack([np.cos(ang2), np.sin(ang2)], 1)
    g1 = rho * np.stack([np.cos(th), np.sin(th)], 1)
    g2 = rho * np.stack([np.cos(th[sigma]), np.sin(th[sigma])], 1)
    return f1, g1, f2, g2, lab, ang1, ang2, th, sigma


def all_gaps(k, swap_i, swap_j, n_per=60, margin=0.6):
    _, _, _, _, lab, ang1, ang2, th, sigma = build_pair(
        k, swap_i, swap_j, n_per, 1.0, margin)
    s1 = np.cos(ang1[:, None] - th[None, :])
    s2 = np.cos(ang2[:, None] - th[sigma][None, :])
    top1 = np.argmax(s1, 1)
    a = s1[np.arange(len(lab)), top1][:, None] - s1
    b = s2[np.arange(len(lab)), top1][:, None] - s2
    off = np.ones_like(a, bool)
    off[np.arange(len(lab)), top1] = False
    return a[off], b[off], a, b, lab


def _std(Z):
    Z = Z - Z.mean(0, keepdims=True)
    s = Z.std(0, keepdims=True)
    s[s < 1e-12] = 1.0
    return Z / s


def d_svd(Z, W):
    Zs, Ws = _std(Z), _std(W)
    C = Zs.T @ Ws / (len(Zs) - 1)
    return 1.0 - np.linalg.svd(C, compute_uv=False).mean()


def mcca(Z, W):
    Zs, Ws = _std(Z), _std(W)
    n = len(Zs)
    Czz = Zs.T @ Zs / (n - 1) + 1e-10 * np.eye(2)
    Cww = Ws.T @ Ws / (n - 1) + 1e-10 * np.eye(2)
    Czw = Zs.T @ Ws / (n - 1)
    iz = np.linalg.inv(np.linalg.cholesky(Czz))
    iw = np.linalg.inv(np.linalg.cholesky(Cww))
    return np.linalg.svd(iz @ Czw @ iw.T, compute_uv=False).mean()


def dfg_pair(k, swap_i, swap_j, n_per=60, margin=0.6, rho=20.0):
    f1, g1, f2, g2, lab, *_ = build_pair(k, swap_i, swap_j, n_per, rho, margin)
    M = 2
    L = (g1 - g1[0])[1:M + 1].T
    Lp = (g2 - g2[0])[1:M + 1].T
    d1 = d_svd(f1 @ L, f2 @ Lp)
    idx = [np.where(lab == j)[0][0] for j in range(M + 1)]
    N = (f1 - f1[idx[0]])[idx[1:]].T
    Np = (f2 - f2[idx[0]])[idx[1:]].T
    d2 = d_svd(g1 @ N, g2 @ Np)
    return max(d1, d2), mcca(f1, f2)


def D_alpha(f, g, f2, g2, alpha):
    lp = f @ g.T
    lp -= logsumexp(lp, 1, keepdims=True)
    lq = f2 @ g2.T
    lq -= logsumexp(lq, 1, keepdims=True)
    if np.isinf(alpha):
        return np.mean(np.max(lp - lq, 1))
    if abs(alpha - 1) < TOL:
        return np.mean(np.sum(np.exp(lp) * (lp - lq), 1))
    return np.mean(logsumexp(alpha * lp + (1 - alpha) * lq, 1) / (alpha - 1))


print("=" * 76)
print("CAN alpha* BE DRIVEN LARGE WHILE REPRESENTATIONS STAY FAR APART?")
print("=" * 76)
print(f"{'k':>4} {'swap':>9} {'alpha*':>9} {'d_fg':>9} {'mCCA':>8} "
      f"{'maxKL':>9} {'sig=max(b-a)':>13}")

rows = []
for k in [7, 12, 20, 32, 48]:
    for dist in [1, 2, 3]:
        if dist >= k:
            continue
        i, j = 0, dist
        aa, bb, _, _, _ = all_gaps(k, i, j, n_per=40, margin=0.6)
        ast = astar(aa, bb)
        d, mc = dfg_pair(k, i, j, n_per=40, margin=0.6, rho=40.0)
        keep = np.abs(aa - bb) > 1e-8
        sig = (bb[keep] - aa[keep]).max() if keep.any() else 0.0
        f1, g1, f2, g2, *_ = build_pair(k, i, j, 40, 40.0, 0.6)
        kl = D_alpha(f1, g1, f2, g2, 1.0)
        rows.append((k, dist, ast, d, mc, kl, sig))
        print(f"{k:>4} {f'0<->{dist}':>9} {ast:>9.3f} {d:>9.4f} {mc:>8.4f} "
              f"{kl:>9.2e} {sig:>13.4f}")

R = np.array([(r[2], r[3]) for r in rows if np.isfinite(r[2])])
if len(R) > 2:
    print(f"\ncorr(alpha*, d_fg) = {np.corrcoef(R[:,0], R[:,1])[0,1]:+.4f}")
    print(f"alpha* range [{R[:,0].min():.2f}, {R[:,0].max():.2f}]   "
          f"d_fg range [{R[:,1].min():.4f}, {R[:,1].max():.4f}]")

# --- does d_fg stay bounded below as alpha* grows? ------------------------
print("\n" + "=" * 76)
print("ASYMPTOTIC: adjacent swap, k increasing (gentlest possible permutation)")
print("=" * 76)
print(f"{'k':>5} {'alpha*':>10} {'d_fg':>10} {'mCCA':>9} {'d_fg*alpha*':>12}")
for k in [8, 16, 32, 64, 128, 256]:
    aa, bb, _, _, _ = all_gaps(k, 0, 1, n_per=24, margin=0.6)
    ast = astar(aa, bb)
    d, mc = dfg_pair(k, 0, 1, n_per=24, margin=0.6, rho=60.0)
    print(f"{k:>5} {ast:>10.3f} {d:>10.5f} {mc:>9.4f} {d*ast:>12.5f}")
