"""
CENTRAL CONJECTURE (resolves items 1 and 2 simultaneously)

  alpha* = min_{x,y: b>a} b_y/(b_y - a_y).  Note b_y - a_y = s_y - s'_y.
  So  alpha* >= A   <=>   |s_y - s'_y| <= b_y / A   for all constraining (x,y).

  Since b_y <= 2 (cosines), this forces a UNIFORM perturbation bound
        sup_{x,y} |s_y - s'_y|  <=  2/A.
  Uniformly close cosine structure forces the standardized projections
  L^T f and L'^T f' to be close, hence

        d_fg  <=  C(M) / (A * sigma),        sigma = spread of s over inputs.

  If true, no family can have alpha* unbounded AND d_fg bounded below:
  item 1 is impossible and item 2 is a theorem. It also yields the missing
  finite-order guarantee: D_alpha small => d_fg <= C/alpha.

Tests below:
  (T1) does alpha* >= A imply sup|s-s'| <= 2/A ?
  (T2) does d_fg * alpha* stay bounded (ideally -> 0) ?
  (T3) does M=3 geometry break the cap on alpha* ?  (predicted: no)
"""

import numpy as np
from scipy.special import logsumexp

TOL = 1e-9


def astar_from(a, b):
    m = b > a + 1e-8
    return np.min(b[m] / (b[m] - a[m])) if m.any() else np.inf


def _std(Z):
    Z = Z - Z.mean(0, keepdims=True)
    s = Z.std(0, keepdims=True)
    s[s < 1e-12] = 1.0
    return Z / s


def d_svd(Z, W):
    Zs, Ws = _std(Z), _std(W)
    C = Zs.T @ Ws / (len(Zs) - 1)
    return 1.0 - np.linalg.svd(C, compute_uv=False).mean()


def measures(f1, g1, f2, g2, lab, M):
    L = (g1 - g1[0])[1:M + 1].T
    Lp = (g2 - g2[0])[1:M + 1].T
    d1 = d_svd(f1 @ L, f2 @ Lp)
    idx = [np.where(lab == j)[0][0] for j in range(M + 1)]
    N = (f1 - f1[idx[0]])[idx[1:]].T
    Np = (f2 - f2[idx[0]])[idx[1:]].T
    d2 = d_svd(g1 @ N, g2 @ Np)
    return max(d1, d2)


def analyse(f1, g1u, f2, g2u, lab, M, rho):
    """f: unit embeddings, gu: UNIT unembedding directions."""
    s1 = f1 @ g1u.T
    s2 = f2 @ g2u.T
    top = np.argmax(s1, 1)
    a = s1[np.arange(len(lab)), top][:, None] - s1
    b = s2[np.arange(len(lab)), top][:, None] - s2
    off = np.ones_like(a, bool)
    off[np.arange(len(lab)), top] = False
    ast = astar_from(a[off], b[off])
    sup_pert = np.abs(s1 - s2).max()
    sigma = (s1 - s1[:, [0]]).std(0)[1:].min()
    d = measures(f1, rho * g1u, f2, rho * g2u, lab, M)
    return ast, sup_pert, d, sigma


# ------------------------------------------------- M=2 circular (reference) --
def circular(k, swap, n_per=40, margin=0.6):
    th = 2 * np.pi * np.arange(k) / k
    sig = np.arange(k)
    sig[[0, swap]] = sig[[swap, 0]]
    phi = np.linspace(-margin, margin, n_per) * np.pi / k
    lab = np.repeat(np.arange(k), n_per)
    a1 = (th[:, None] + phi[None, :]).ravel()
    a2 = (th[sig][:, None] + phi[None, :]).ravel()
    f1 = np.stack([np.cos(a1), np.sin(a1)], 1)
    f2 = np.stack([np.cos(a2), np.sin(a2)], 1)
    g1 = np.stack([np.cos(th), np.sin(th)], 1)
    g2 = np.stack([np.cos(th[sig]), np.sin(th[sig])], 1)
    return f1, g1, f2, g2, lab, 2


# ------------------------------------- M=3, Fibonacci sphere, non-uniform ----
def sphere(k, swap, n_per=12, jitter=0.0, seed=0, nonuniform=False):
    """Unembedding directions quasi-uniform on S^2; optional norm heterogeneity."""
    rng = np.random.default_rng(seed)
    i = np.arange(k) + 0.5
    z = 1 - 2 * i / k
    r = np.sqrt(np.maximum(0, 1 - z * z))
    ph = np.pi * (1 + 5 ** 0.5) * i
    G = np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)
    if jitter:
        G = G + jitter * rng.normal(size=G.shape)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    if nonuniform:
        G = G * (1 + 0.5 * rng.random((k, 1)))
        G /= np.linalg.norm(G, axis=1, keepdims=True)

    sig = np.arange(k)
    sig[[0, swap]] = sig[[swap, 0]]

    # embeddings: small random perturbations around each unembedding direction
    F1, F2, lab = [], [], []
    for j in range(k):
        d = rng.normal(size=(n_per, 3)) * 0.18
        v1 = G[j] + d
        v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
        # model 2: same offset applied around the RELABELLED direction
        R = _align(G[j], G[sig[j]])
        v2 = v1 @ R.T
        F1.append(v1); F2.append(v2); lab.append(np.full(n_per, j))
    return (np.vstack(F1), G, np.vstack(F2), G[sig], np.concatenate(lab), 3)


def _align(u, v):
    """Rotation taking u to v (Rodrigues)."""
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    c = float(u @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        w = np.eye(3)[np.argmin(np.abs(u))]
        ax = np.cross(u, w); ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * K @ K
    ax = np.cross(u, v); s = np.linalg.norm(ax); ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


# ------------------------------------------------------------------- driver --
print("=" * 82)
print("T1/T2  does alpha* >= A force sup|s-s'| <= 2/A, and d_fg <~ C/A ?")
print("=" * 82)
print(f"{'geom':>14} {'alpha*':>8} {'sup|s-s|':>9} {'2/a*':>8} {'T1 ok':>6} "
      f"{'d_fg':>8} {'d_fg*a*':>9}")

rows = []
for k in [7, 12, 20, 32]:
    for sw in [1, 2, 3]:
        f1, g1, f2, g2, lab, M = circular(k, sw)
        ast, sp, d, sg = analyse(f1, g1, f2, g2, lab, M, 40.0)
        ok = "yes" if sp <= 2 / ast + 1e-9 else "NO"
        rows.append((ast, sp, d))
        print(f"{f'circ k={k} s={sw}':>14} {ast:>8.3f} {sp:>9.4f} "
              f"{2/ast:>8.4f} {ok:>6} {d:>8.4f} {d*ast:>9.4f}")

print()
for k in [12, 24, 40]:
    for sw in [1, 3]:
        for nu in [False, True]:
            f1, g1, f2, g2, lab, M = sphere(k, sw, nonuniform=nu)
            ast, sp, d, sg = analyse(f1, g1, f2, g2, lab, M, 40.0)
            ok = "yes" if sp <= 2 / ast + 1e-9 else "NO"
            rows.append((ast, sp, d))
            tag = f"S2 k={k} s={sw}{'*' if nu else ''}"
            print(f"{tag:>14} {ast:>8.3f} {sp:>9.4f} {2/ast:>8.4f} {ok:>6} "
                  f"{d:>8.4f} {d*ast:>9.4f}")

R = np.array(rows)
print(f"\nT1 violations: {(R[:,1] > 2/R[:,0] + 1e-9).sum()} / {len(R)}")
print(f"T2  max d_fg*alpha* = {(R[:,2]*R[:,0]).max():.4f}   "
      f"max alpha* = {R[:,0].max():.3f}")

print("\n" + "=" * 82)
print("T3  can M=3 + jitter + non-uniform norms push alpha* past the M=2 cap?")
print("=" * 82)
best = 0.0
for k in [8, 16, 30, 50]:
    for jit in [0.0, 0.05, 0.15]:
        for sd in range(4):
            for sw in [1, 2, 5]:
                if sw >= k:
                    continue
                f1, g1, f2, g2, lab, M = sphere(k, sw, jitter=jit, seed=sd,
                                                nonuniform=True)
                ast, sp, d, sg = analyse(f1, g1, f2, g2, lab, M, 40.0)
                if np.isfinite(ast):
                    best = max(best, ast)
print(f"max alpha* over all M=3 configurations tried: {best:.4f}")
print(f"(M=2 circular cap observed earlier: ~1.24)")
