"""
inspect_one.py -- what a SINGLE trained model can tell you.

All pairwise measures need >= 2 models, but the confidence profile that decides
Remark 16 (and hence how Theorem 14 should be sold) is essentially a per-model
quantity. So one run answers the highest-priority open question.

    python inspect_one.py runs/d2_s0_sm0.0.npz
"""

import sys
import numpy as np
from scipy.special import logsumexp


def main(path):
    z = np.load(path)
    lp = z['logp'].astype(np.float64)
    y = z['y']
    M = int(z['dim'])
    N, C = lp.shape
    p = np.exp(lp)
    ps = np.sort(p, 1)[:, ::-1]

    print("=" * 70)
    print(f"{path}   N={N}  C={C}  M={M}  smooth={float(z['smooth'])}")
    print("=" * 70)
    print(f"test accuracy      {float(z['acc']):.4f}")
    print(f"test NLL           {float(z['loss']):.4f}")

    print("\n-- sanity ---------------------------------------------------")
    ok = abs(np.exp(logsumexp(lp, 1)) - 1).max()
    print(f"log-probs normalised (max |sum p - 1|)   {ok:.2e}  "
          f"{'OK' if ok < 1e-5 else 'BAD'}")
    print(f"classes present in labels                {len(np.unique(y))}/{C}")
    print(f"classes ever predicted                   "
          f"{len(np.unique(lp.argmax(1)))}/{C}")
    print(f"embedding rank (needs {M} for diversity)  "
          f"{np.linalg.matrix_rank(z['f'].astype(np.float64))}")
    print(f"unembedding rank                         "
          f"{np.linalg.matrix_rank(z['g'].astype(np.float64))}")

    print("\n-- confidence -----------------------------------------------")
    ent = -(p * np.clip(lp, -700, 0)).sum(1)
    print(f"mean max-probability   {ps[:, 0].mean():.6f}")
    print(f"median max-probability {np.median(ps[:, 0]):.6f}")
    print(f"mean entropy (nats)    {ent.mean():.4f}   (uniform = {np.log(C):.4f})")
    print(f"mean perplexity        {np.exp(ent).mean():.3f}  "
          f"(effective no. of live classes)")

    print("\n-- C1: delta_m profile (THE question) -----------------------")
    print(f"  {'m':>3} {'delta_m':>13} {'1/delta_m':>12} "
          f"{'median m-th prob':>17}")
    for m in range(2, C + 1):
        dm = ps[:, :m].min()
        print(f"  {m:>3} {dm:>13.3e} {1/max(dm,1e-300):>12.3e} "
              f"{np.median(ps[:, m-1]):>17.3e}")
    d_need = ps[:, :M + 1].min()
    print(f"\n  M+1 = {M+1} labels are needed for diversity, so the operative"
          f" floor is\n  delta_{M+1} = {d_need:.3e},  giving 1/delta = "
          f"{1/max(d_need,1e-300):.3e}")

    print("\n-- what this does to Theorem 14 -----------------------------")
    print("  modulus is 4 sqrt(2 lambda_max) sqrt(eps) / (delta_m psi_min);")
    print("  with delta_m above, KL would have to reach")
    for tgt in [1.0, 0.1]:
        eps = (tgt * d_need / 8.0) ** 2
        print(f"    eps < {eps:.3e}   for the bound to fall below {tgt}"
              f"  (taking lambda_max~2, psi_min~1)")
    print("\n  READING: if these eps are far below any KL you observe between"
          "\n  retrained models, Thm 14 is vacuous here and Rem. 16 stands."
          "\n  If delta_{M+1} is only ~1e-3 or larger, Rem. 16 is too"
          "\n  pessimistic and that section needs rewriting.")

    print("\n-- next -----------------------------------------------------")
    print("  pairwise measures need >= 2 models. Minimum useful run:")
    print("    for S in 0 1 2 3; do python train.py --dim 2 --seed $S"
          " --out runs; done")
    print("    python analyze.py --runs runs --out results/cifar")
    print("  4 seeds = 6 pairs, enough for a first read; 10 seeds = 45 pairs"
          " for the paper.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1])
