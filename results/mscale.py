"""
Option 3 executed: does representation dimension M lift alpha* without paying
in representational dissimilarity?

This bears directly on the open problem. If alpha*_m grows without bound in M
while d_fg stays bounded away from 0, then no finite order controls on Theta
and the open problem is settled negatively.

We track the PARETO FRONTIER of (alpha*_m, d_fg) rather than the max of either
alone, because maximising alpha* alone is trivially achieved by making the two
models nearly identical.
"""

import numpy as np
from a3 import make_pair, gaps_and_margin, alpha_star_m, dfg

MFLOOR = 0.08
DFLOOR = 0.10          # require genuinely dissimilar representations

print("=" * 80)
print(f"M-SCALING   alpha*_m at margin floor {MFLOOR}, subject to d_fg >= {DFLOOR}")
print("=" * 80)
print(f"{'M':>3} {'trials':>7} {'best a*_m':>10} {'d_fg there':>11} "
      f"{'median a*_m':>12} {'frontier a* @ d>=0.3':>21}")

rows = []
for M in [2, 3, 4, 6, 8, 12]:
    best, bestd, vals, front = 0.0, 0.0, [], 0.0
    ntr = 0
    for trial in range(260):
        rng = np.random.default_rng(97 * M + 3 * trial)
        k = int(rng.choice([M + 4, 2 * M + 4, 3 * M + 6, 26]))
        if k <= M + 1:
            continue
        sd = int(rng.choice([1, 2, 3]))
        sp = float(rng.choice([0.10, 0.16, 0.24, 0.32]))
        irr = bool(rng.random() < 0.5)
        nu = bool(rng.random() < 0.35)
        try:
            F1, G, F2, Gp, lab = make_pair(M, k, 20, rng, spread=sp,
                                           irregular=irr, nonuniform=nu,
                                           swap_dist=sd)
            a, b, off, margin, ok = gaps_and_margin(F1, G, F2, Gp)
            if ok.mean() < 0.9:
                continue
            A, ns = alpha_star_m(a, b, off, margin, ok, MFLOOR)
            if ns < 20 or not np.isfinite(A):
                continue
            d = dfg(F1, G, F2, Gp, lab, M)
            ntr += 1
            if d >= DFLOOR:
                vals.append(A)
                if A > best:
                    best, bestd = A, d
            if d >= 0.30:
                front = max(front, A)
        except Exception:
            continue
    med = np.median(vals) if vals else np.nan
    rows.append((M, best, bestd, med, front))
    print(f"{M:>3} {ntr:>7} {best:>10.4f} {bestd:>11.4f} {med:>12.4f} "
          f"{front:>21.4f}")

R = np.array([(r[0], r[1]) for r in rows if r[1] > 0])
if len(R) > 2:
    lg = np.polyfit(np.log(R[:, 0]), R[:, 1], 1)[0]
    print(f"\nbest alpha*_m vs log M: slope = {lg:+.4f}")
    print(f"growth from M=2 to M={int(R[-1,0])}: "
          f"{R[0,1]:.3f} -> {R[-1,1]:.3f}  ({R[-1,1]/R[0,1]:.2f}x)")

print("\nreading: if alpha*_m saturates in M, the open problem likely resolves")
print("POSITIVELY (some finite order controls). If it grows without bound while")
print("the d_fg>=0.3 frontier keeps pace, it resolves NEGATIVELY.")
