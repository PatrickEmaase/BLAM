"""
Is the M=2 obstruction TOPOLOGICAL or METRIC?

TOPOLOGICAL HYPOTHESIS.  On S^1 the unembedding directions carry a cyclic
order. A permutation sigma either
  (a) preserves cyclic order -- sigma(i) = (+-i + b) mod k, the dihedral group
      D_k -- in which case it is realised by a rotation or reflection, i.e. a
      LINEAR map, so d_fg = 0; or
  (b) breaks cyclic order, in which case realising it by an isotopy of the
      circle forces a COLLISION: some input sees two labels tie under model 1
      while model 2 ranks them oppositely, giving a_y -> 0 with b_y > 0 and
      hence alpha* -> 1.
Either way no sigma gives both alpha* large and d_fg large. On S^{M-1} with
M >= 3 there is no cyclic order (point-configuration braid groups are trivial
in dimension >= 3), so permutations are realisable without collisions.

METRIC ALTERNATIVE.  With non-uniform norms the tie condition in M=2 becomes
lambda_i cos th_i = lambda_j cos th_j, which is not a reflection symmetry. If
non-uniform norms lift alpha* for order-BREAKING permutations, the obstruction
is metric, not topological.

Decisive test: condition on cyclic-order class and compare.
"""

import numpy as np
from itertools import permutations
from harness import make_pair, gaps_and_margin, alpha_star_m, dfg

MF = 0.08


def is_dihedral(s, k):
    s = np.asarray(s)
    for a in (1, -1):
        for b in range(k):
            if np.all(s == (a * np.arange(k) + b) % k):
                return True
    return False


def breaks(s, k):
    """Adjacency violations of the cyclic order: 0 iff dihedral."""
    s = np.asarray(s)
    n = 0
    for i in range(k):
        if (s[(i + 1) % k] - s[i]) % k not in (1, k - 1):
            n += 1
    return n


def evaluate(M, k, sigma, nu, rng, irregular=False, n=2400):
    F, G, Fp, Gp, lab, sg, U, lam = make_pair(
        M, k, n, rng, sigma=sigma, irregular=irregular, nonuniform=nu)
    a, b, off, margin, ok = gaps_and_margin(F, G, Fp, Gp)
    if ok.mean() < 0.6:
        return None
    A, ns = alpha_star_m(a, b, off, margin, ok, MF)
    d = dfg(F, G, Fp, Gp, lab, M)
    if ns < 30 or not np.isfinite(d):
        return None
    return A, d, ok.mean()


# ---------------------------------------------- M=2, all permutations, k=6 ---
k = 6
print("=" * 78)
print(f"M=2, ALL {__import__("math").factorial(k)} permutations of k={k}, by cyclic-order class")
print("=" * 78)

for nu in [False, True]:
    dih = dict(A=[], d=[])
    brk = dict(A=[], d=[])
    for s in permutations(range(k)):
        rng = np.random.default_rng(11)
        r = evaluate(2, k, list(s), nu, rng)
        if r is None:
            continue
        A, d, _ = r
        tgt = dih if is_dihedral(s, k) else brk
        tgt['A'].append(min(A, 50)); tgt['d'].append(d)
    tag = "non-uniform norms" if nu else "uniform norms"
    print(f"\n{tag}")
    for nm, g in [("order-preserving (dihedral)", dih),
                  ("order-breaking", brk)]:
        if not g['A']:
            print(f"  {nm:<28} none")
            continue
        A = np.array(g['A']); d = np.array(g['d'])
        both = ((A > 1.3) & (d > 0.15)).sum()
        print(f"  {nm:<28} n={len(A):>3}  "
              f"max a*={A.max():>6.2f}  med a*={np.median(A):>5.2f}  "
              f"max d_fg={d.max():>5.3f}  med d_fg={np.median(d):>5.3f}  "
              f"both-large={both}")

# ----------------------------------- does alpha* fall with order violations --
print("\n" + "=" * 78)
print("alpha*_m vs number of cyclic-order violations (M=2, k=8)")
print("=" * 78)
k = 8
for nu in [False, True]:
    buckets = {}
    rng0 = np.random.default_rng(5)
    for _ in range(600):
        s = rng0.permutation(k)
        nb = breaks(s, k)
        r = evaluate(2, k, s, nu, np.random.default_rng(11))
        if r is None:
            continue
        A, d, _ = r
        buckets.setdefault(nb, []).append((min(A, 50), d))
    tag = "non-uniform" if nu else "uniform   "
    print(f"\nnorms = {tag}   {'breaks':>7} {'n':>5} {'med a*':>8} "
          f"{'max a*':>8} {'med d_fg':>9}")
    for nb in sorted(buckets):
        v = np.array(buckets[nb])
        print(f"{'':>19} {nb:>7} {len(v):>5} {np.median(v[:,0]):>8.3f} "
              f"{v[:,0].max():>8.3f} {np.median(v[:,1]):>9.4f}")

# -------------------------------------------------- M=3 control, same sigmas -
print("\n" + "=" * 78)
print("M=3 control: order-breaking permutations, uniform norms")
print("=" * 78)
k = 8
A3, d3 = [], []
rng0 = np.random.default_rng(5)
for _ in range(120):
    s = rng0.permutation(k)
    if is_dihedral(s, k):
        continue
    r = evaluate(3, k, s, False, np.random.default_rng(11))
    if r is None:
        continue
    A3.append(min(r[0], 50)); d3.append(r[1])
A3, d3 = np.array(A3), np.array(d3)
print(f"n={len(A3)}  med a*={np.median(A3):.3f}  max a*={A3.max():.3f}  "
      f"med d_fg={np.median(d3):.4f}  both-large={(((A3>1.3)&(d3>0.15)).sum())}")
