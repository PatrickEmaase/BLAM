import sys, glob, itertools
import numpy as np
from scipy.special import logsumexp

def _s(Z):
    Z = Z - Z.mean(0); s = Z.std(0); s[s < 1e-12] = 1; return Z / s
def dsvd(Z, W):
    C = _s(Z).T @ _s(W) / (len(Z) - 1)
    return float(1 - np.linalg.svd(C, compute_uv=False).mean())
def dfg(f1, g1, f2, g2, y, M):
    ys = list(range(M + 1)); L = (g1 - g1[0])[ys[1:]].T; Lp = (g2 - g2[0])[ys[1:]].T
    idx = [int(np.where(y == j)[0][0]) for j in ys]
    N = (f1 - f1[idx[0]])[idx[1:]].T; Np = (f2 - f2[idx[0]])[idx[1:]].T
    return max(dsvd(f1 @ L, f2 @ Lp), dsvd(g1 @ N, g2 @ Np))

paths = sys.argv[1:] or sorted(glob.glob('runs/cifar/*sm0.0.npz'))
R = {}
for p in paths:
    z = np.load(p)
    if float(z['smooth']) != 0.0: continue
    R.setdefault(int(z['dim']), []).append(z)

print(f"{'M':>3} {'n':>4} {'pairs':>6} {'lam_max':>8} {'ceiling':>8} "
      f"{'med d_fg':>9} {'binds?':>7} | {'old':>7} {'new':>7} "
      f"{'nonvac':>7} {'viol':>5}")
for M in sorted(R):
    G = R[M]
    old, new, dd, lams = [], [], [], []
    for A, B in itertools.combinations(G, 2):
        lp = A['logp'].astype(float); lq = B['logp'].astype(float)
        z1 = lp[:, 1:M+1] - lp[:, [0]]; z2 = lq[:, 1:M+1] - lq[:, [0]]
        s1, s2 = z1.std(0), z2.std(0); sd = (z1 - z2).std(0); tau = abs(s1 - s2)
        pl = np.minimum(s1, s2); Z = _s(z1)
        lam = float(np.linalg.eigvalsh(Z.T @ Z / (len(Z)-1)).max()); lams.append(lam)
        old.append(2*np.sqrt(lam)*sd.max()/pl.min())
        new.append(float(np.sqrt(lam)*np.sqrt(np.mean(((tau+sd)/pl)**2))))
        dd.append(dfg(A['f'].astype(float), A['g'].astype(float),
                      B['f'].astype(float), B['g'].astype(float),
                      A['y'], M))
    old, new, dd = map(np.array, (old, new, dd))
    lam = float(np.median(lams)); ceil = 1/(2*lam); md = float(np.median(dd))
    print(f"{M:>3} {len(G):>4} {len(dd):>6} {lam:>8.3f} {ceil:>8.3f} "
          f"{md:>9.4f} {'YES' if md>=ceil else 'no':>7} | "
          f"{np.median(old):>7.3f} {np.median(new):>7.3f} "
          f"{100*(new<1).mean():>6.0f}% {int((dd>new+1e-9).sum()):>5}")
print("\n'binds' = median d_fg above the Prop-13 ceiling => the cap, not the")
print("constant, is what blocks certification at that dimension.")
