"""
CI regression suite. Run before any experiment result enters the paper.

Guards the two error classes that actually bit us:
  R1  silent filtering  -- any gate that rejects >50% of configurations fails
  R2  asymmetric alpha* -- alpha*_sym = inf must imply d_fg = 0, to O(1/n)
  R3  the rate lemma must hold in BOTH directions
  R4  proved constants must not be violated
"""

import sys
import numpy as np
from harness import make_pair, gaps_and_margin, dfg, d_svd
from factorial import astar_sym, astar_dir
from perx import one_x, D_alpha_exact, rate_pred

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


print("R1  acceptance rate of the shared-argmax gate")
acc = tot = 0
for tr in range(40):
    rng = np.random.default_rng(600 + tr)
    k = int(rng.choice([8, 12, 18]))
    F, G, Fp, Gp, lab, *_ = make_pair(
        2, k, 1500, rng, irregular=bool(tr % 2), nonuniform=bool(tr % 3))
    _, _, _, _, ok = gaps_and_margin(F, G, Fp, Gp)
    tot += 1
    acc += ok.mean() >= 0.6
check("acceptance >= 50%", acc / tot >= 0.5, f"({acc}/{tot} = {acc/tot:.0%})")

print("\nR2  alpha*_sym = inf  =>  d_fg < 5/n")
worst = 0.0
for n in [1000, 4000, 16000]:
    for b in range(1, 4):
        k = 7
        rng = np.random.default_rng(11)
        s = (np.arange(k) + b) % k
        F, G, Fp, Gp, lab, *_ = make_pair(2, k, n, rng, sigma=list(s))
        a, bb, off, mar, ok = gaps_and_margin(F, G, Fp, Gp)
        A, *_ = astar_sym(a, bb, off, mar, ok, 0.08)
        if not np.isinf(A):
            continue
        d = abs(dfg(F, G, Fp, Gp, lab, 2))
        worst = max(worst, d * n / 5.0)
check("d_fg < 5/n for all n", worst <= 1.0, f"(worst ratio {worst:.3f})")

print("\nR3  rate lemma verifies in both directions")
k = 7
sg = np.random.default_rng(1).permutation(k)
aa, bb, A, B = one_x(k, 0.25 * np.pi / 7, 0, sg)
errs = []
for rev in [False, True]:
    p, q = (B, A) if rev else (A, B)
    pred = astar_dir(bb, aa) if rev else astar_dir(aa, bb)
    lo, hi = 1.0, 30.0
    for _ in range(50):
        m = .5 * (lo + hi)
        if D_alpha_exact(p, q, 160, m) > D_alpha_exact(p, q, 50, m):
            hi = m
        else:
            lo = m
    errs.append(abs(.5 * (lo + hi) - pred) / pred)
check("crossover rel err < 1e-3 both dirs", max(errs) < 1e-3,
      f"(fwd {errs[0]:.2e}, rev {errs[1]:.2e})")

print("\nR4  proved constants hold")
v12 = v17 = tot4 = 0
for M in [2, 3, 4]:
    for tr in range(8):
        rng = np.random.default_rng(31 * M + tr)
        k = int(rng.choice([8, 12, 18]))
        F, G, Fp, Gp, lab, *_ = make_pair(
            M, k, 3000, rng, sigma=list(rng.permutation(k)), shrink=0.2)
        a, b, off, mar, ok = gaps_and_margin(F, G, Fp, Gp)
        if ok.mean() < 0.6:
            continue
        As, *_ = astar_sym(a, b, off, mar, ok, 0.08)
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
        tot4 += 1
        v12 += d > 2 * np.sqrt(lm) * rho2 / psi + 1e-9
        if np.isfinite(As) and d > 1e-3:
            Delta = float(np.where(off, a, 0).max())
            v17 += d > 4 * np.sqrt(M) * Delta / (psi * As) + 1e-9
check("Thm 12 (lambda_max form)", v12 == 0, f"({v12}/{tot4} violations)")
check("Thm 17 (graded control)", v17 == 0, f"({v17}/{tot4} violations)")

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
