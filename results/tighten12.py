"""
Closing the 2.6x gap in Thm 12.

The chain is  divergence -> rho_2/psi_min -> (1 - rho_bar) -> d_SVD.
Splitting it shows the slack is NOT in the spectral step.

LEMMA A (spectral, tight).  In standardised coordinates
Var[z'_l - w'_l] = 2(1 - rho_l) EXACTLY, with rho_l the l-th diagonal entry of
Sigma_{z'w'}. So sum_l Var = 2M(1 - rho_bar) exactly, and

    d_SVD <= ||E||_F / sqrt(M) <= sqrt(2 lambda_max (1 - rho_bar)).      (A)

Since tr <= nuclear norm gives 1 - rho_bar >= d_SVD, (A) meets the
Prop-13 floor sqrt(2 lambda_max d_SVD) with equality whenever the trace
attains the nuclear norm. There is nothing to gain here.

LEMMA B (divergence -> correlation, loose).  The old argument bounds
    Var[z'_l - w'_l] <= (2 rho_2 / psi_min)^2
with rho_2 = max_l std(Delta_l), i.e. it takes the WORST coordinate and
applies it to all M of them. Replacing the max by a root-mean-square,

    R^2 := (1/M) sum_l std(Delta_l)^2 / min(s_l, s'_l)^2,      R <= rho_2/psi_min

gives the same form with R in place of rho_2/psi_min:

    d_SVD <= 2 sqrt(lambda_max) R.                                       (B)

Measured below: (B) is the free win, (A) is the ceiling.
"""

import numpy as np
from harness import make_pair, gaps_and_margin, d_svd


def quantities(M, k, rng, **kw):
    F, G, Fp, Gp, lab, *_ = make_pair(M, k, 4000, rng, **kw)
    a, b, off, mar, ok = gaps_and_margin(F, G, Fp, Gp)
    if ok.mean() < 0.6:
        return None
    t1, t2 = F @ G.T, Fp @ Gp.T
    lp = t1 - np.log(np.exp(t1).sum(1, keepdims=True))
    lq = t2 - np.log(np.exp(t2).sum(1, keepdims=True))
    z1 = lp[:, 1:M + 1] - lp[:, [0]]
    z2 = lq[:, 1:M + 1] - lq[:, [0]]

    s1, s2 = z1.std(0), z2.std(0)
    psi = min(s1.min(), s2.min())
    sd = (z1 - z2).std(0)
    rho2 = sd.max()

    # standardised correlation diagonal
    Z = (z1 - z1.mean(0)) / s1
    W = (z2 - z2.mean(0)) / s2
    rho_l = (Z * W).mean(0)
    rho_bar = float(rho_l.mean())
    lam = float(np.linalg.eigvalsh(Z.T @ Z / (len(Z) - 1)).max())

    R = float(np.sqrt(np.mean(sd ** 2 / np.minimum(s1, s2) ** 2)))
    d = d_svd(z1, z2)
    return dict(d=d, lam=lam, rho2=rho2, psi=psi, R=R, rho_bar=rho_bar,
                old=2 * np.sqrt(lam) * rho2 / psi,
                rms=2 * np.sqrt(lam) * R,
                spec=np.sqrt(2 * lam * max(1 - rho_bar, 0)),
                floor=np.sqrt(2 * lam * d))


print("=" * 82)
print("WHERE THE SLACK LIVES")
print("=" * 82)
print(f"{'M':>3} {'n':>4} {'d_SVD':>8} {'old bnd':>9} {'RMS (B)':>9} "
      f"{'spec (A)':>9} {'floor':>8} | {'old/RMS':>8} {'RMS/spec':>9} "
      f"{'spec/floor':>11}")
agg = {}
for M in [2, 3, 4, 6]:
    rows = []
    for tr in range(40):
        rng = np.random.default_rng(1500 + 17 * M + tr)
        k = int(rng.choice([8, 12, 18, 26]))
        q = quantities(M, k, rng, sigma=list(rng.permutation(k)),
                       irregular=bool(tr % 2), nonuniform=bool(tr % 3),
                       shrink=float(rng.choice([0.0, 0.3, 0.6])))
        if q and q['d'] > 1e-6:
            rows.append(q)
    med = {key: float(np.median([r[key] for r in rows])) for key in rows[0]}
    agg[M] = (rows, med)
    print(f"{M:>3} {len(rows):>4} {med['d']:>8.4f} {med['old']:>9.4f} "
          f"{med['rms']:>9.4f} {med['spec']:>9.4f} {med['floor']:>8.4f} | "
          f"{med['old']/med['rms']:>8.3f} {med['rms']/med['spec']:>9.3f} "
          f"{med['spec']/med['floor']:>11.3f}")

print("\nvalidity (no bound may fall below d_SVD):")
for M, (rows, _) in agg.items():
    v = {k: sum(r['d'] > r[k] + 1e-9 for r in rows)
         for k in ('old', 'rms', 'spec')}
    print(f"  M={M}: violations old {v['old']}, RMS {v['rms']}, "
          f"spectral {v['spec']}  / {len(rows)}")

print("\nnon-vacuity (bound < 1):")
for M, (rows, _) in agg.items():
    nv = {k: np.mean([r[k] < 1 for r in rows]) for k in ('old', 'rms', 'spec')}
    print(f"  M={M}: old {nv['old']:.0%}, RMS {nv['rms']:.0%}, "
          f"spectral {nv['spec']:.0%}")

print("\nprojected effect on the CIFAR-10 numbers "
      "(d_fg=0.128, lambda_max=1.411, old bound=1.558):")
r = float(np.median([agg[2][1]['old'] / agg[2][1]['rms'],
                     agg[3][1]['old'] / agg[3][1]['rms']]))
print(f"  median old/RMS gain at M=2,3 is {r:.2f}x  ->  bound "
      f"{1.558/r:.3f}  ({'NON-VACUOUS' if 1.558/r < 1 else 'still vacuous'})")
print(f"  spectral form would give sqrt(2*1.411*(1-rho_bar)); with "
      f"1-rho_bar = d_fg it is {np.sqrt(2*1.411*0.128):.3f}")
