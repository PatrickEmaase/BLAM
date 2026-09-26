"""
Corrected numerics under alpha*_sym, plus a regression test for the dichotomy.

DICHOTOMY (Thm 1). alpha*_sym = inf  <=>  a_y(x) = b_y(x) for all x,y
  <=>  <f(x),g(y)> - <f'(x),g'(y)> = c(x) independent of y
  <=>  p_{f,g} = p_{f',g'}   (c(x) cancels in the softmax normaliser)
  <=>  (f,g) ~_L (f',g')     (linear identifiability, under diversity).

REGRESSION TEST: alpha*_sym = inf  =>  d_fg < 1e-9.
"""

import numpy as np
from itertools import permutations
from harness import make_pair, gaps_and_margin, dfg
from factorial import astar_sym, astar_dir, is_dihedral

MF = 0.08

# ------------------------------------------- 1. dichotomy regression test ----
print("=" * 78)
print("1. DICHOTOMY REGRESSION:  alpha*_sym = inf  =>  d_fg = 0")
print("=" * 78)
viol, ninf, nfin, maxd_inf, mind_fin = 0, 0, 0, 0.0, 9.9
for k in [6, 7, 8]:
    rng0 = np.random.default_rng(4)
    pool = [np.arange(k)] + [(a * np.arange(k) + b) % k
                             for a in (1, -1) for b in range(k)]
    pool += [rng0.permutation(k) for _ in range(40)]
    for s in pool:
        for nu in [False, True]:
            rng = np.random.default_rng(11)
            F, G, Fp, Gp, lab, sg, U, lam = make_pair(
                2, k, 2000, rng, sigma=list(s), nonuniform=nu)
            a, b, off, margin, ok = gaps_and_margin(F, G, Fp, Gp)
            if ok.mean() < 0.6:
                continue
            As, af, ar, ns = astar_sym(a, b, off, margin, ok, MF)
            d = dfg(F, G, Fp, Gp, lab, 2)
            if ns < 30 or not np.isfinite(d) or np.isnan(As):
                continue
            if np.isinf(As):
                ninf += 1
                maxd_inf = max(maxd_inf, abs(d))
                if abs(d) > 1e-9:
                    viol += 1
            else:
                nfin += 1
                mind_fin = min(mind_fin, abs(d))
print(f"alpha*_sym = inf : n={ninf:>4}   max |d_fg| = {maxd_inf:.3e}")
print(f"alpha*_sym < inf : n={nfin:>4}   min |d_fg| = {mind_fin:.3e}")
print(f"VIOLATIONS of (alpha*_sym=inf => d_fg=0): {viol}")

# ------------------------------------- 2. Table 1 geometry, both directions --
print("\n" + "=" * 78)
print("2. SINGLE-INPUT GEOMETRY: forward / reverse / symmetric crossover")
print("=" * 78)
from perx import one_x, D_alpha_exact
k, sigma = 7, np.random.default_rng(1).permutation(7)
phi = 0.25 * np.pi / 7
aa, bb, A, B = one_x(k, phi, 0, sigma)
f_ = astar_dir(aa, bb); r_ = astar_dir(bb, aa)
print(f"alpha*_fwd = {f_:.6f}   alpha*_rev = {r_:.6f}   "
      f"alpha*_sym = {min(f_, r_):.6f}")


def bisect(A_, B_, rev=False):
    lo, hi = 1.0, 30.0
    for _ in range(50):
        m = .5 * (lo + hi)
        p, q = (B_, A_) if rev else (A_, B_)
        if D_alpha_exact(p, q, 160, m) > D_alpha_exact(p, q, 50, m):
            hi = m
        else:
            lo = m
    return .5 * (lo + hi)


bf, br = bisect(A, B), bisect(A, B, rev=True)
print(f"bisected  fwd = {bf:.6f} (err {abs(bf-f_)/f_:.2e})   "
      f"rev = {br:.6f} (err {abs(br-r_)/r_:.2e})")

# ---------------------------------------------- 3. M-scaling under sym -------
print("\n" + "=" * 78)
print("3. M-SCALING under alpha*_sym  (margin floor 0.08, d_fg >= 0.10)")
print("=" * 78)
print(f"{'M':>3} {'n':>5} {'best a*_sym':>12} {'d_fg there':>11} {'median':>9}")
msc = []
for M in [2, 3, 4, 6, 8, 12]:
    best, bd, vals = 0.0, 0.0, []
    for trial in range(200):
        rng = np.random.default_rng(97 * M + 3 * trial)
        k = int(rng.choice([M + 4, 2 * M + 4, 26]))
        if k <= M + 1:
            continue
        F, G, Fp, Gp, lab, sg, U, lam = make_pair(
            M, k, 1800, rng, irregular=bool(rng.random() < .5),
            nonuniform=bool(rng.random() < .35),
            swap_dist=int(rng.choice([1, 2, 3])),
            shrink=float(rng.choice([0.0, 0.25, 0.5])))
        a, b, off, margin, ok = gaps_and_margin(F, G, Fp, Gp)
        if ok.mean() < 0.6:
            continue
        As, af, ar, ns = astar_sym(a, b, off, margin, ok, MF)
        if ns < 30 or np.isnan(As) or np.isinf(As):
            continue
        d = dfg(F, G, Fp, Gp, lab, M)
        if not np.isfinite(d) or d < 0.10:
            continue
        vals.append(As)
        if As > best:
            best, bd = As, d
    msc.append((M, len(vals), best, bd, np.median(vals) if vals else np.nan))
    print(f"{M:>3} {len(vals):>5} {best:>12.3f} {bd:>11.3f} "
          f"{(np.median(vals) if vals else np.nan):>9.3f}")
print(f"\nCEILING over all non-equivalent pairs: "
      f"{max(r[2] for r in msc):.4f}")

# ------------------------------------- 4. panel (c) curves under sym ---------
curves = []
for M, kk, c in [(2, 12, "#d1495b"), (3, 14, "#1b6ca8"), (4, 16, "#2e4057")]:
    rng = np.random.default_rng(3)
    F, G, Fp, Gp, lab, sg, U, lam = make_pair(M, kk, 4000, rng, shrink=0.2)
    a, b, off, margin, ok = gaps_and_margin(F, G, Fp, Gp)
    fin = margin[np.isfinite(margin)]
    xs, ys = [], []
    for m in np.linspace(0, np.percentile(fin, 90), 20):
        As, _, _, ns = astar_sym(a, b, off, margin, ok, m)
        if ns >= 30 and np.isfinite(As):
            xs.append(m / fin.max()); ys.append(As)
    curves.append((xs, ys, c, "-", f"$M={M}$"))
np.save("panelc.npy", np.array(curves, dtype=object), allow_pickle=True)
print(f"\npanel (c) under sym: peaks "
      f"{[round(max(cv[1]), 3) for cv in curves]}")
