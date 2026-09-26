"""
Track A1-A2: divergence ladder + rate verification on the norm-growth family.

Constructed models (Appendix F.1 geometry, M=2):
  model 1: g(y_j) = rho * (cos th_j, sin th_j),  th_j = 2*pi*j/k
           f(x)   = (cos(th_j + phi), sin(th_j + phi)) for x in cluster j
  model 2: relabel by permutation sigma; cluster j is rotated to sit at
           th_{sigma(j)} with the SAME angular offset phi, so the top label and
           its cosine are preserved (assumptions iii-iv).

Per input x in cluster j, with s_y = cos(f(x), g(y)):
  a_y = s_{yhat} - s_y   (gap, model 1)
  b_y = s'_{yhat} - s'_y (gap, model 2)
  logits = c * rho * s,  so p(y|x) ∝ exp(-c*rho*a_y).

Predicted exponent for the Renyi integrand at label y:
  E_alpha(y) = alpha*a_y + (1-alpha)*b_y = b_y + alpha*(a_y - b_y)
  alpha*(y)  = b_y / (b_y - a_y)   whenever b_y > a_y
  alpha*     = min over x, y
  kappa(alpha) = min over y of min{E_alpha(y), a_y, b_y}   (decay rate of D_alpha)
"""

import numpy as np
from scipy.special import logsumexp

# ----------------------------------------------------------------- geometry --

def build(k, n_per, rho, seed=0, sigma=None, c=1.0):
    rng = np.random.default_rng(seed)
    th = 2 * np.pi * np.arange(k) / k
    if sigma is None:
        sigma = np.arange(k)

    # angular offsets within each cluster, kept inside +-pi/k so the intended
    # label is strictly the argmax
    phi = rng.uniform(-0.8 * np.pi / k, 0.8 * np.pi / k, size=(k, n_per))

    lab = np.repeat(np.arange(k), n_per)
    ang1 = (th[:, None] + phi).ravel()                 # model 1 embedding angle
    ang2 = (th[sigma][:, None] + phi).ravel()          # model 2 embedding angle

    f1 = c * np.stack([np.cos(ang1), np.sin(ang1)], 1)
    f2 = c * np.stack([np.cos(ang2), np.sin(ang2)], 1)

    g1 = rho * np.stack([np.cos(th), np.sin(th)], 1)
    # model 2: unembedding of label y sits at angle th[sigma[y]]
    g2 = rho * np.stack([np.cos(th[sigma]), np.sin(th[sigma])], 1)
    return f1, g1, f2, g2, lab, ang1, ang2, th, sigma


def cosines(k, ang, th_used):
    """s[x, y] = cos(angle(f(x)) - th_used[y])."""
    return np.cos(ang[:, None] - th_used[None, :])


def gaps(k, n_per, seed, sigma, c=1.0):
    """Return a, b arrays of shape (N, k), rho-independent."""
    _, _, _, _, lab, ang1, ang2, th, sigma = build(k, n_per, 1.0, seed, sigma, c)
    s1 = cosines(k, ang1, th)            # model 1
    s2 = cosines(k, ang2, th[sigma])     # model 2
    top = lab
    a = s1[np.arange(len(lab)), top][:, None] - s1
    b = s2[np.arange(len(lab)), top][:, None] - s2
    return a, b, lab


def logp(f, g, c_rho_scale=1.0):
    z = f @ g.T
    return z - logsumexp(z, axis=1, keepdims=True)


# --------------------------------------------------------------- divergences --

def renyi(lp, lq, alpha):
    """E_x[ D_alpha( p(.|x) || q(.|x) ) ], natural log."""
    if np.isinf(alpha):
        return np.mean(np.max(lp - lq, axis=1))
    if abs(alpha - 1.0) < 1e-12:
        p = np.exp(lp)
        return np.mean(np.sum(p * (lp - lq), axis=1))
    t = alpha * lp + (1 - alpha) * lq
    return np.mean(logsumexp(t, axis=1) / (alpha - 1))


def tv(lp, lq):
    return np.mean(0.5 * np.sum(np.abs(np.exp(lp) - np.exp(lq)), axis=1))


def hellinger2(lp, lq):
    return np.mean(np.sum((np.exp(0.5 * lp) - np.exp(0.5 * lq)) ** 2, axis=1))


def jsd(lp, lq):
    lm = np.logaddexp(lp, lq) - np.log(2)
    p, q = np.exp(lp), np.exp(lq)
    return np.mean(0.5 * np.sum(p * (lp - lm), 1) + 0.5 * np.sum(q * (lq - lm), 1))


# ------------------------------------------------ representational measures --

def _std_cols(Z):
    Z = Z - Z.mean(0, keepdims=True)
    s = Z.std(0, keepdims=True)
    s[s < 1e-12] = 1.0
    return Z / s


def d_svd(Z, W):
    """1 - mean PLS-SVD covariance between standardized Z, W."""
    Zs, Ws = _std_cols(Z), _std_cols(W)
    n = Zs.shape[0]
    C = Zs.T @ Ws / (n - 1)
    sv = np.linalg.svd(C, compute_uv=False)
    return 1.0 - sv.mean()


def mcca(Z, W):
    Zs, Ws = _std_cols(Z), _std_cols(W)
    n = Zs.shape[0]
    Czz = Zs.T @ Zs / (n - 1) + 1e-10 * np.eye(Zs.shape[1])
    Cww = Ws.T @ Ws / (n - 1) + 1e-10 * np.eye(Ws.shape[1])
    Czw = Zs.T @ Ws / (n - 1)
    iz = np.linalg.inv(np.linalg.cholesky(Czz))
    iw = np.linalg.inv(np.linalg.cholesky(Cww))
    return np.linalg.svd(iz @ Czw @ iw.T, compute_uv=False).mean()


def d_fg(f1, g1, f2, g2, lab, k):
    """max{ d_SVD(L^T f, L'^T f'), d_SVD(N^T g, N'^T g') } with M=2."""
    M = f1.shape[1]
    g1_0 = g1 - g1[0]
    g2_0 = g2 - g2[0]
    L = g1_0[1:M + 1].T          # columns g0(y_i), i=1..M
    Lp = g2_0[1:M + 1].T
    Z1, Z2 = f1 @ L, f2 @ Lp

    # pivot input + M more, one per distinct cluster
    idx = [np.where(lab == j)[0][0] for j in range(M + 1)]
    f1_0 = f1 - f1[idx[0]]
    f2_0 = f2 - f2[idx[0]]
    N = f1_0[idx[1:]].T
    Np = f2_0[idx[1:]].T
    W1, W2 = g1 @ N, g2 @ Np
    return max(d_svd(Z1, Z2), d_svd(W1, W2)), mcca(f1, f2)


# ------------------------------------------------------- simplified LLV (t1) --

def llv_t1(lp, lq):
    """max over y != y0 of sqrt(Var_x[ logp(y|x)/psi(y;p) - logq(y|x)/psi(y;q) ]),
    plus the y0 term, with pivot y0 = 0. No pivot search (A1 only needs a
    rho-robust reference quantity)."""
    y0 = 0
    d_p = lp - lp[:, [y0]]
    d_q = lq - lq[:, [y0]]
    psi_p = d_p.std(0)
    psi_q = d_q.std(0)
    best = 0.0
    for y in range(1, lp.shape[1]):
        if psi_p[y] < 1e-9 or psi_q[y] < 1e-9:
            continue
        v1 = np.std(lp[:, y] / psi_p[y] - lq[:, y] / psi_q[y])
        v2 = np.std(lp[:, y0] / psi_p[y] - lq[:, y0] / psi_q[y])
        best = max(best, v1, v2)
    return best


# ------------------------------------------------------------------- driver --

def predicted(k, n_per, seed, sigma, alphas, c=1.0):
    a, b, lab = gaps(k, n_per, seed, sigma, c)
    off = np.ones_like(a, bool)
    off[np.arange(len(lab)), lab] = False
    aa, bb = a[off], b[off]

    mask = bb > aa
    astar = np.min(bb[mask] / (bb[mask] - aa[mask])) if mask.any() else np.inf

    kap = {}
    for al in alphas:
        if np.isinf(al):
            kap[al] = -np.inf          # D_inf grows like rho
            continue
        E = bb + al * (aa - bb)
        kap[al] = min(E.min(), aa.min(), bb.min())
    return astar, kap, aa.min(), bb.min()


def run(k=7, n_per=120, seed=0, perm_seed=1, rhos=(3, 6, 9, 12, 15, 18, 21, 24)):
    rng = np.random.default_rng(perm_seed)
    sigma = rng.permutation(k)
    while np.all(sigma == np.arange(k)):
        sigma = rng.permutation(k)

    alphas = [0.5, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, np.inf]
    astar, kap, amin, bmin = predicted(k, n_per, seed, sigma, alphas)

    print(f"k={k}  sigma={sigma.tolist()}")
    print(f"predicted alpha* = {astar:.4f}   (min gaps: a={amin:.4f} b={bmin:.4f})\n")

    rows = []
    for rho in rhos:
        f1, g1, f2, g2, lab, _, _, _, _ = build(k, n_per, rho, seed, sigma)
        lp, lq = logp(f1, g1), logp(f2, g2)
        rec = {"rho": rho}
        for al in alphas:
            rec[al] = renyi(lp, lq, al)
        rec["TV"] = tv(lp, lq)
        rec["H2"] = hellinger2(lp, lq)
        rec["JS"] = jsd(lp, lq)
        rec["LLV"] = llv_t1(lp, lq)
        dfg, mc = d_fg(f1, g1, f2, g2, lab, k)
        rec["dfg"], rec["mCCA"] = dfg, mc
        rec["delta"] = np.exp(lp.min())
        rows.append(rec)

    hdr = ["rho", "KL", "TV", "H2", "JS", "D_inf", "LLV", "dfg", "mCCA", "delta"]
    print(" ".join(f"{h:>10}" for h in hdr))
    for r in rows:
        vals = [r["rho"], r[1.0], r["TV"], r["H2"], r["JS"], r[np.inf],
                r["LLV"], r["dfg"], r["mCCA"], r["delta"]]
        print(" ".join(f"{v:>10.3e}" if i else f"{v:>10.0f}"
                       for i, v in enumerate(vals)))

    # ---- A2: measured vs predicted exponential decay rate --------------------
    print("\nA2  decay rates  (fit log|D| ~ -c*rho*kappa over the last 4 rho)")
    print(f"{'alpha':>8} {'kappa_pred':>12} {'kappa_fit':>12} {'D@rho_max':>12} {'status':>10}")
    R = np.array([r["rho"] for r in rows], float)
    for al in alphas:
        y = np.array([r[al] for r in rows], float)
        tail = slice(-4, None)
        lab_s = "grows" if y[-1] > y[0] else "decays"
        if np.all(y[tail] > 0) and np.all(np.isfinite(y[tail])):
            sl = np.polyfit(R[tail], np.log(y[tail]), 1)[0]
            kfit = -sl
        else:
            kfit = np.nan
        kp = kap[al]
        name = "inf" if np.isinf(al) else f"{al:g}"
        kps = "-inf" if np.isinf(kp) else f"{kp:.4f}"
        print(f"{name:>8} {kps:>12} {kfit:>12.4f} {y[-1]:>12.3e} {lab_s:>10}")

    # ---- crossover: smallest alpha on the grid that does not decay ----------
    grow = [al for al in alphas
            if np.array([r[al] for r in rows])[-1]
            > np.array([r[al] for r in rows])[0]]
    print(f"\nmeasured crossover in (grid): first non-decaying alpha = "
          f"{min(grow) if grow else None}   predicted alpha* = {astar:.4f}")

    # ---- sufficiency bound check -------------------------------------------
    print("\nsufficiency check   max_y r_y  <=  D_alpha + log(1/delta)/(alpha-1)")
    print(f"{'rho':>5} {'max_r':>10} {'delta':>10} " +
          " ".join(f"{'a=' + str(int(a)):>12}" for a in [3, 8, 20]))
    for r, rho in zip(rows, rhos):
        f1, g1, f2, g2, lab, _, _, _, _ = build(k, n_per, rho, seed, sigma)
        lp, lq = logp(f1, g1), logp(f2, g2)
        maxr = np.mean(np.max(lp - lq, 1))
        dlt = np.exp(lp.min())
        bnds = [renyi(lp, lq, a) + np.log(1 / dlt) / (a - 1) for a in [3, 8, 20]]
        print(f"{rho:>5} {maxr:>10.3f} {dlt:>10.2e} " +
              " ".join(f"{b:>12.3f}" for b in bnds))
    return rows, astar, kap


if __name__ == "__main__":
    run()
