"""
measures.py -- all quantities the paper reports, pure numpy.

Works from saved arrays, so training runs once and analysis is cheap and
re-runnable. Every model dumps an .npz with:
    f     (N, M)   test-set embeddings
    g     (C, M)   unembeddings
    logp  (N, C)   log conditional probabilities
    y     (N,)     test labels
    loss, acc, seed, dim, smooth
"""

import numpy as np
from scipy.special import logsumexp

EPS = 1e-12


# ------------------------------------------------------------ divergences ---
def d_kl(lp, lq):
    """KL with the 0 log 0 = 0 convention; returns inf if supp(p) escapes supp(q)."""
    p = np.exp(lp)
    if np.any((p > 0) & ~np.isfinite(lq)):
        return float('inf')
    m = p > 0
    d = np.zeros_like(lp)
    d[m] = lp[m] - lq[m]          # evaluate only on the support, avoids inf-inf
    return float(np.mean(np.sum(p * d, 1)))


def d_tv(lp, lq):
    return float(np.mean(0.5 * np.abs(np.exp(lp) - np.exp(lq)).sum(1)))


def d_renyi(lp, lq, alpha):
    if np.isinf(alpha):
        return float(np.mean(np.max(lp - lq, 1)))
    if abs(alpha - 1) < 1e-12:
        return d_kl(lp, lq)
    return float(np.mean(logsumexp(alpha * lp + (1 - alpha) * lq, 1) / (alpha - 1)))


def d_hellinger2(lp, lq):
    return float(np.mean(np.sum((np.exp(.5 * lp) - np.exp(.5 * lq)) ** 2, 1)))


def d_js(lp, lq):
    lm = np.logaddexp(lp, lq) - np.log(2)
    p, q = np.exp(lp), np.exp(lq)
    return float(np.mean(.5 * (p * (lp - lm)).sum(1) + .5 * (q * (lq - lm)).sum(1)))


# ------------------------------------------- representational (dis)similarity
def _std(Z):
    Z = Z - Z.mean(0, keepdims=True)
    s = Z.std(0, keepdims=True)
    s[s < EPS] = 1.0
    return Z / s


def d_svd(Z, W):
    C = _std(Z).T @ _std(W) / (len(Z) - 1)
    return float(1.0 - np.linalg.svd(C, compute_uv=False).mean())


def m_cca(Z, W):
    Zs, Ws = _std(Z), _std(W)
    n, M = Zs.shape
    A = np.linalg.inv(np.linalg.cholesky(Zs.T @ Zs / (n - 1) + 1e-9 * np.eye(M)))
    B = np.linalg.inv(np.linalg.cholesky(Ws.T @ Ws / (n - 1) + 1e-9 * np.eye(M)))
    return float(np.linalg.svd(A @ (Zs.T @ Ws / (n - 1)) @ B.T,
                               compute_uv=False).mean())


def linear_cka(Z, W):
    Zc = Z - Z.mean(0); Wc = W - W.mean(0)
    num = np.linalg.norm(Zc.T @ Wc, 'fro') ** 2
    den = np.linalg.norm(Zc.T @ Zc, 'fro') * np.linalg.norm(Wc.T @ Wc, 'fro')
    return float(num / max(den, EPS))


def procrustes(Z, W):
    """Normalised orthogonal-Procrustes distance in [0, 1]."""
    Zc = Z - Z.mean(0); Wc = W - W.mean(0)
    Zc /= max(np.linalg.norm(Zc, 'fro'), EPS)
    Wc /= max(np.linalg.norm(Wc, 'fro'), EPS)
    s = np.linalg.svd(Zc.T @ Wc, compute_uv=False).sum()
    return float(max(0.0, 2 - 2 * s) / 2)


def d_fg(f1, g1, f2, g2, y, M, label_subset=None, input_idx=None):
    """Representational dissimilarity of Def. 4.6, on a chosen label subset."""
    ys = list(range(M + 1)) if label_subset is None else list(label_subset)
    assert len(ys) >= M + 1, "need M+1 labels for diversity"
    piv, rest = ys[0], ys[1:M + 1]
    L = (g1 - g1[piv])[rest].T
    Lp = (g2 - g2[piv])[rest].T
    if input_idx is None:
        input_idx = [int(np.where(y == j)[0][0]) for j in ys[:M + 1]]
    N = (f1 - f1[input_idx[0]])[input_idx[1:M + 1]].T
    Np = (f2 - f2[input_idx[0]])[input_idx[1:M + 1]].T
    return max(d_svd(f1 @ L, f2 @ Lp), d_svd(g1 @ N, g2 @ Np))


# ------------------------------------------------- confidence + alpha* -------
def delta_profile(lp, lq, m_max=None):
    """delta_m = min over x of the m-th largest prob, jointly over both models."""
    C = lp.shape[1]
    m_max = m_max or C
    a = np.sort(np.exp(lp), 1)[:, ::-1]
    b = np.sort(np.exp(lq), 1)[:, ::-1]
    return {m: float(min(a[:, :m].min(), b[:, :m].min()))
            for m in range(2, m_max + 1)}


def alpha_star_sym(lp, lq, margin_floor=0.0):
    """Symmetric critical order. Uses logit gaps recovered from log-probs."""
    top1 = lp.argmax(1); top2 = lq.argmax(1)
    ok = top1 == top2
    n = np.arange(len(lp))
    a = lp[n, top1][:, None] - lp
    b = lq[n, top2][:, None] - lq
    off = np.ones_like(a, bool); off[n, top1] = False
    margin = np.where(off, a, np.inf).min(1)
    sel = ok & (margin >= margin_floor)
    if sel.sum() < 5:
        return np.nan, 0, float(ok.mean())
    A, B = a[sel][off[sel]], b[sel][off[sel]]

    def one(P, Q):
        mk = Q > P + 1e-8
        return float(np.min(Q[mk] / (Q[mk] - P[mk]))) if mk.any() else np.inf
    return min(one(A, B), one(B, A)), int(sel.sum()), float(ok.mean())


def thm12_bound(lp, lq, M, label_subset=None):
    """Refined: d_fg <= sqrt(lambda_max) * Q,  Q^2 = mean_l ((tau_l+sd_l)/psi_l)^2."""
    ys = list(range(M + 1)) if label_subset is None else list(label_subset)
    piv, rest = ys[0], ys[1:M + 1]
    z1 = lp[:, rest] - lp[:, [piv]]
    z2 = lq[:, rest] - lq[:, [piv]]
    s1, s2 = z1.std(0), z2.std(0)
    sd = (z1 - z2).std(0)
    tau = np.abs(s1 - s2)
    psi_l = np.minimum(s1, s2)
    Zc = _std(z1)
    lm = float(np.linalg.eigvalsh(Zc.T @ Zc / (len(z1) - 1)).max())
    Q = float(np.sqrt(np.mean(((tau + sd) / psi_l) ** 2)))
    return np.sqrt(lm) * Q, float(sd.max()), float(psi_l.min()), lm


def _restrict(lp, idx):
    out = np.full_like(lp, -np.inf)
    np.put_along_axis(out, idx, np.take_along_axis(lp, idx, 1), 1)
    return out - logsumexp(out, 1, keepdims=True)


def topk_pair(lp, lq, k):
    """Restrict BOTH models to the reference model's top-k support (C1).

    Restricting each to its own support makes the KL ill-defined whenever the
    supports differ, which they generically do. The reference support is taken
    from lp; swap the arguments for the other direction.
    """
    idx = np.argsort(-lp, 1)[:, :k]
    return _restrict(lp, idx), _restrict(lq, idx)


def tail_pair(lp, lq, k):
    """Restrict BOTH models to everything below the reference top-k (C1)."""
    idx = np.argsort(-lp, 1)[:, k:]
    return _restrict(lp, idx), _restrict(lq, idx)
