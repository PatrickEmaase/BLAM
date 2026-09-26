"""
Is Thm 12's non-vacuity capped structurally, like p_(M+1) <= 1/(M+1) capped
Thm 14?  YES, and the argument is three lines.

Standardised coordinates give Var[z'_l - w'_l] = 2(1 - rho_l) with
rho_l = (Sigma_{z'w'})_{ll}. Hence

    sum_l Var[z'_l - w'_l] = 2M - 2 trace(Sigma_{z'w'}).

For any square matrix, trace <= nuclear norm = sum of singular values, and
sum_i sigma_i = M * mSVD = M (1 - d_SVD). Therefore

    sum_l Var[z'_l - w'_l] >= 2M - 2M(1 - d_SVD) = 2M d_SVD.          (*)

Our bound is d_SVD <= sqrt(lambda_max * sum_l Var) / sqrt(M), so by (*)

    bound >= sqrt(2 lambda_max d_SVD),

and bound < 1 FORCES  d_SVD < 1 / (2 lambda_max)  <=  1/2.

So the bound can only ever certify pairs that are ALREADY similar. The 5%
non-vacuity rate is not a loose constant; it is the fraction of pairs whose
d_fg happens to fall below a hard ceiling.

Same question for Thm 18 (graded control): bound = 4 sqrt(M) Delta /
(psi_min A) is below 1 only when A > 4 sqrt(M) Delta / psi_min. Since
alpha*_sym <= ~1.35 empirically, this needs Delta/psi_min < 0.34/sqrt(M).
Measured below.
"""

import numpy as np
from harness import make_pair, gaps_and_margin, d_svd, dfg
from factorial import astar_sym

print("=" * 76)
print("CEILING TEST 1:  does  bound < 1  force  d_SVD < 1/(2 lambda_max) ?")
print("=" * 76)
print(f"{'M':>3} {'n':>5} {'max d_SVD among':>17} {'ceiling':>9} "
      f"{'viol':>5} {'min bound/sqrt(2 lam d)':>24}")
print(f"{'':>3} {'':>5} {'non-vacuous pairs':>17}")
for M in [2, 3, 4, 6]:
    ds, bs, lms, ratio = [], [], [], []
    for tr in range(60):
        rng = np.random.default_rng(700 + 13 * M + tr)
        k = int(rng.choice([8, 12, 18, 26]))
        F, G, Fp, Gp, lab, *_ = make_pair(
            M, k, 3000, rng, sigma=list(rng.permutation(k)),
            irregular=bool(tr % 2), nonuniform=bool(tr % 3),
            shrink=float(rng.choice([0.0, 0.3, 0.6])))
        a, b, off, mar, ok = gaps_and_margin(F, G, Fp, Gp)
        if ok.mean() < 0.6:
            continue
        t1, t2 = F @ G.T, Fp @ Gp.T
        lp = t1 - np.log(np.exp(t1).sum(1, keepdims=True))
        lq = t2 - np.log(np.exp(t2).sum(1, keepdims=True))
        z1 = lp[:, 1:M + 1] - lp[:, [0]]
        z2 = lq[:, 1:M + 1] - lq[:, [0]]
        rho2 = (z1 - z2).std(0).max()
        psi = min(z1.std(0).min(), z2.std(0).min())
        Zc = (z1 - z1.mean(0)) / np.maximum(z1.std(0), 1e-12)
        lm = float(np.linalg.eigvalsh(Zc.T @ Zc / (len(z1) - 1)).max())
        d = d_svd(z1, z2)
        bnd = 2 * np.sqrt(lm) * rho2 / psi
        ds.append(d); bs.append(bnd); lms.append(lm)
        if d > 1e-9:
            ratio.append(bnd / np.sqrt(2 * lm * d))
    ds, bs, lms = np.array(ds), np.array(bs), np.array(lms)
    nv = bs < 1
    ceil = 1.0 / (2 * np.median(lms))
    viol = int((np.array(ratio) < 1 - 1e-9).sum())
    mx = ds[nv].max() if nv.any() else np.nan
    print(f"{M:>3} {len(ds):>5} {mx:>17.4f} {ceil:>9.4f} {viol:>5} "
          f"{(min(ratio) if ratio else np.nan):>24.4f}")
print("\n  'viol' counts violations of bound >= sqrt(2 lambda_max d_SVD);")
print("  the last column being >= 1 confirms the inequality empirically.")

print("\n" + "=" * 76)
print("CEILING TEST 2:  is Cor 19's cap ever attainable?")
print("  needs alpha > 4 sqrt(M) Delta / psi_min, but alpha*_sym <= ~1.35")
print("=" * 76)
print(f"{'M':>3} {'n':>5} {'median Delta/psi':>17} {'required alpha':>15} "
      f"{'attainable?':>12}")
for M in [2, 3, 4, 6]:
    rr = []
    for tr in range(40):
        rng = np.random.default_rng(900 + 11 * M + tr)
        k = int(rng.choice([8, 12, 18]))
        F, G, Fp, Gp, lab, *_ = make_pair(
            M, k, 3000, rng, sigma=list(rng.permutation(k)), shrink=0.3)
        a, b, off, mar, ok = gaps_and_margin(F, G, Fp, Gp)
        if ok.mean() < 0.6:
            continue
        t1, t2 = F @ G.T, Fp @ Gp.T
        lp = t1 - np.log(np.exp(t1).sum(1, keepdims=True))
        lq = t2 - np.log(np.exp(t2).sum(1, keepdims=True))
        z1 = lp[:, 1:M + 1] - lp[:, [0]]
        z2 = lq[:, 1:M + 1] - lq[:, [0]]
        psi = min(z1.std(0).min(), z2.std(0).min())
        Delta = float(np.where(off, a, 0).max())
        rr.append(Delta / psi)
    med = float(np.median(rr))
    req = 4 * np.sqrt(M) * med
    print(f"{M:>3} {len(rr):>5} {med:>17.3f} {req:>15.1f} "
          f"{'NO' if req > 1.35 else 'yes':>12}")
print("\n  required alpha far above the empirical alpha*_sym ceiling of ~1.35")
print("  => Cor 19 is finite and explicit but numerically out of reach too.")
