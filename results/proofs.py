"""
(1) Regression test, corrected: is the residual d_fg in the alpha*_sym = inf
    class estimation noise? If so it must fall like n^{-1/2}.

(2) Validation of the constants derived for Thms 12, 16, 17.

    Thm 12/16.  Let z1 = L^T f(x), z2 = L'^T f'(x), so
      z1_i - z2_i = r_{y_i}(x) - r_{y_0}(x),  r_y := log p(y|x) - log p'(y|x).
    With s_l = std_x(z1_l) = psi_x(y_l;p) and delta_l := z1_l - z2_l,
      |s_l - s'_l| <= std(delta_l),
      sqrt(Var[z'_l - w'_l]) <= 2 std(delta_l) / psi_min.
    Lemma E.7 with Hoffman-Wielandt gives
      d_SVD <= sqrt(sum_l Var[z'_l - w'_l]) <= 2 sqrt(M) rho_2 / psi_min,
    where rho_2 := max_l std(delta_l).  (Weyl alone costs an extra sqrt(M).)

    Thm 17.  alpha*_sym >= A  =>  sup|delta_l| <= 2 Delta / A, hence
      d_fg <= 4 sqrt(M) Delta / (psi_min A).

    COROLLARY (the ceiling becomes a theorem):
      alpha_0(d_0) := sup{alpha*_sym : d_fg >= d_0} <= 4 sqrt(M) Delta /
      (psi_min d_0).
"""

import numpy as np
from harness import make_pair, gaps_and_margin, dfg, d_svd
from factorial import astar_sym

MF = 0.08


# ------------------------------------------------ (1) noise convergence -----
print("=" * 78)
print("(1) REGRESSION TEST: is residual d_fg in the inf-class estimation noise?")
print("=" * 78)
print(f"{'n samples':>10} {'max|d_fg| (inf class)':>23} {'x sqrt(n)':>11}")
k = 7
rot = [(np.arange(k) + b) % k for b in range(1, 4)]
for n in [500, 1000, 2000, 4000, 8000, 16000]:
    mx = 0.0
    for s in rot:
        for nu in [False, True]:
            rng = np.random.default_rng(11)
            F, G, Fp, Gp, lab, sg, U, lam = make_pair(
                2, k, n, rng, sigma=list(s), nonuniform=nu)
            a, b, off, margin, ok = gaps_and_margin(F, G, Fp, Gp)
            As, *_ = astar_sym(a, b, off, margin, ok, MF)
            if not np.isinf(As):
                continue
            d = dfg(F, G, Fp, Gp, lab, 2)
            if np.isfinite(d):
                mx = max(mx, abs(d))
    print(f"{n:>10} {mx:>23.3e} {mx*np.sqrt(n):>11.4f}")
print("flat right-hand column => residual is O(n^-1/2) sampling noise, not bias")


# ------------------------------------- (2) validate the derived constants ---
def quantities(M, k, sigma, nu, irregular, shrink, seed=11, n=4000):
    rng = np.random.default_rng(seed)
    F, G, Fp, Gp, lab, sg, U, lam = make_pair(
        M, k, n, rng, sigma=sigma, irregular=irregular,
        nonuniform=nu, shrink=shrink)
    a, b, off, margin, ok = gaps_and_margin(F, G, Fp, Gp)
    if ok.mean() < 0.6:
        return None
    As, af, ar, ns = astar_sym(a, b, off, margin, ok, MF)
    if ns < 30 or np.isnan(As):
        return None
    # log-ratio structure at rho = 1 (logits are already the inner products)
    t1 = F @ G.T
    t2 = Fp @ Gp.T
    lp = t1 - np.log(np.exp(t1).sum(1, keepdims=True))
    lq = t2 - np.log(np.exp(t2).sum(1, keepdims=True))
    r = lp - lq
    z1 = lp[:, 1:M + 1] - lp[:, [0]]
    z2 = lq[:, 1:M + 1] - lq[:, [0]]
    delta = z1 - z2
    rho2 = delta.std(0).max()
    psi = min(z1.std(0).min(), z2.std(0).min())
    Delta = float(np.where(off, a, 0).max())
    d = d_svd(z1, z2)
    return As, d, rho2, psi, Delta, r


print("\n" + "=" * 78)
print("(2a) Thm 12/16 constant:  d_SVD <= 2 sqrt(M) rho_2 / psi_min ?")
print("=" * 78)
print(f"{'M':>3} {'k':>4} {'d_SVD':>9} {'bound':>10} {'ratio':>8} {'ok':>4}")
viol = 0; tot = 0; ratios = []
for M in [2, 3, 4]:
    for k in [8, 12, 18]:
        for tr in range(6):
            rng = np.random.default_rng(31 * M + 5 * k + tr)
            sg = rng.permutation(k)
            q = quantities(M, k, list(sg), bool(tr % 2), False, 0.2)
            if q is None:
                continue
            As, d, rho2, psi, Delta, r = q
            bnd = 2 * np.sqrt(M) * rho2 / psi
            tot += 1; ratios.append(d / bnd)
            if d > bnd + 1e-9:
                viol += 1
            if tr == 0:
                print(f"{M:>3} {k:>4} {d:>9.4f} {bnd:>10.4f} "
                      f"{d/bnd:>8.4f} {'yes' if d <= bnd else 'NO':>4}")
print(f"\nviolations: {viol}/{tot}   median tightness d/bound = "
      f"{np.median(ratios):.4f}   max = {np.max(ratios):.4f}")

print("\n" + "=" * 78)
print("(2b) Thm 17:  alpha*_sym >= A  =>  d_fg <= 4 sqrt(M) Delta/(psi_min A)")
print("     and the corollary  alpha_0(d_0) <= 4 sqrt(M) Delta/(psi_min d_0)")
print("=" * 78)
viol2 = 0; tot2 = 0; slack = []
for M in [2, 3, 4]:
    for k in [8, 12, 18]:
        for tr in range(8):
            rng = np.random.default_rng(101 * M + 7 * k + tr)
            sg = rng.permutation(k)
            q = quantities(M, k, list(sg), bool(tr % 2), bool(tr % 3), 0.2)
            if q is None:
                continue
            As, d, rho2, psi, Delta, r = q
            if np.isinf(As) or d < 1e-3:
                continue
            tot2 += 1
            bnd = 4 * np.sqrt(M) * Delta / (psi * As)       # Thm 17
            if d > bnd + 1e-9:
                viol2 += 1
            cap = 4 * np.sqrt(M) * Delta / (psi * d)        # corollary
            slack.append(cap / As)
print(f"Thm 17 violations: {viol2}/{tot2}")
print(f"corollary cap / measured alpha*_sym:  median {np.median(slack):.1f}x, "
      f"min {np.min(slack):.2f}x  (>=1 means the cap holds)")
print(f"corollary violations: {(np.array(slack) < 1).sum()}/{len(slack)}")
