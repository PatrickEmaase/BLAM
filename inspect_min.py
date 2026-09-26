import sys
import numpy as np
from scipy.special import logsumexp

z = np.load(sys.argv[1])
lp = z['logp'].astype(np.float64); y = z['y']; M = int(z['dim'])
N, C = lp.shape
p = np.exp(lp); ps = np.sort(p, 1)[:, ::-1]
ent = -(p * np.clip(lp, -700, 0)).sum(1)

print(f"N={N} C={C} M={M} smooth={float(z['smooth'])}")
print(f"acc {float(z['acc']):.4f}   NLL {float(z['loss']):.4f}")
print(f"normalised: max|sum p -1| = {abs(np.exp(logsumexp(lp,1))-1).max():.1e}")
print(f"classes predicted {len(np.unique(lp.argmax(1)))}/{C}   "
      f"rank(f)={np.linalg.matrix_rank(z['f'].astype(float))} "
      f"rank(g)={np.linalg.matrix_rank(z['g'].astype(float))} (need {M})")
print(f"mean max-prob {ps[:,0].mean():.6f}   mean entropy {ent.mean():.4f} "
      f"(uniform {np.log(C):.4f})   mean perplexity {np.exp(ent).mean():.3f}")
print("\n  m      delta_m     1/delta_m   median m-th prob")
for m in range(2, C + 1):
    d = ps[:, :m].min()
    print(f"{m:>3} {d:>12.3e} {1/max(d,1e-300):>13.3e} {np.median(ps[:,m-1]):>15.3e}")
d = ps[:, :M+1].min()
print(f"\noperative floor delta_{M+1} = {d:.3e}  ->  1/delta = {1/max(d,1e-300):.3e}")
for t in (1.0, 0.1):
    print(f"  Thm 14 needs KL < {(t*d/8.0)**2:.3e} for its bound to fall below {t}")
