"""
(iii) SHARPENING Thm 12.

The loose step is the trace bound. With E_{kj} = Cov[z'_k, Delta'_j],
    ||E||_F^2 = sum_j sum_k Cov[z'_k, Delta'_j]^2,
and sum_k Cov[z'_k, Delta'_j]^2 is the squared norm of the projection of
Delta'_j onto span{z'_k}. Bounding it by lambda_max(Corr(z')) Var[Delta'_j]
rather than M Var[Delta'_j] gives

    d_SVD <= 2 sqrt(lambda_max) rho_2 / psi_min,      lambda_max <= M,

with equality only when the standardised coordinates are perfectly correlated.

(i) delta_m REFORMULATION of Thm 14. Only labels in Y_LLV enter the bound, and
Y_LLV may be any diverse subset, so delta may be evaluated on a high-mass
subset Y_m. No tail term appears; the constraint is only |Y_m| >= M+1.
"""

import numpy as np
from harness import make_pair, gaps_and_margin, d_svd


def lam_max_corr(Z):
    Zc = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)
    C = Zc.T @ Zc / (len(Z) - 1)
    return float(np.linalg.eigvalsh(C).max())


print("=" * 78)
print("(iii) SHARPENED CONSTANT:  2 sqrt(lambda_max) vs 2 sqrt(M)")
print("=" * 78)
print(f"{'M':>3} {'k':>4} {'lam_max':>8} {'d_SVD':>8} {'old bnd':>9} "
      f"{'new bnd':>9} {'gain':>6} {'ok':>4}")
gains, v_old, v_new, n = [], 0, 0, 0
for M in [2, 3, 4, 6]:
    for k in [8, 12, 18]:
        for tr in range(5):
            rng = np.random.default_rng(31 * M + 5 * k + tr)
            F, G, Fp, Gp, lab, sg, U, lam = make_pair(
                M, k, 4000, rng, sigma=list(rng.permutation(k)),
                irregular=bool(tr % 2), shrink=0.2)
            a, b, off, margin, ok = gaps_and_margin(F, G, Fp, Gp)
            if ok.mean() < 0.6:
                continue
            t1, t2 = F @ G.T, Fp @ Gp.T
            lp = t1 - np.log(np.exp(t1).sum(1, keepdims=True))
            lq = t2 - np.log(np.exp(t2).sum(1, keepdims=True))
            z1 = lp[:, 1:M + 1] - lp[:, [0]]
            z2 = lq[:, 1:M + 1] - lq[:, [0]]
            rho2 = (z1 - z2).std(0).max()
            psi = min(z1.std(0).min(), z2.std(0).min())
            lm = lam_max_corr(z1)
            d = d_svd(z1, z2)
            ob = 2 * np.sqrt(M) * rho2 / psi
            nb = 2 * np.sqrt(lm) * rho2 / psi
            n += 1
            v_old += d > ob + 1e-9
            v_new += d > nb + 1e-9
            gains.append(ob / nb)
            if tr == 0:
                print(f"{M:>3} {k:>4} {lm:>8.3f} {d:>8.4f} {ob:>9.3f} "
                      f"{nb:>9.3f} {ob/nb:>6.2f}x {'yes' if d <= nb else 'NO':>4}")
print(f"\nn={n}   violations: old {v_old}, new {v_new}")
print(f"median improvement factor: {np.median(gains):.2f}x   "
      f"max {np.max(gains):.2f}x")

print("\n" + "=" * 78)
print("(i) delta_m ON A CONFIDENT 10-CLASS CLASSIFIER")
print("=" * 78)
rng = np.random.default_rng(0)
for gap in [6.0, 9.0, 12.0]:
    logits = np.sort(rng.normal(size=(4000, 10)) * 1.0, 1)[:, ::-1]
    logits[:, 0] += gap
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ps = np.sort(p, 1)[:, ::-1]
    print(f"\nlogit gap {gap:.0f}  (mean max-prob {ps[:,0].mean():.4f})")
    print(f"  {'m':>3} {'delta_m':>11} {'tail mass eta_m':>16} "
          f"{'bound factor 1/delta_m':>23}")
    for m in [3, 4, 5, 7, 10]:
        dm = ps[:, :m].min()
        eta = ps[:, m:].sum(1).max() if m < 10 else 0.0
        print(f"  {m:>3} {dm:>11.3e} {eta:>16.3e} {1/dm:>23.3e}")
print("\nM=2 needs only |Y_m| >= 3, so delta_3 is the operative floor.")
print("Gain over delta_10 is only ~17x here: the subset fix helps by about one")
print("order of magnitude, NOT enough to make Thm 14 useful for confident")
print("models. Measure the real profile on CIFAR-10 (experiment C1).")
