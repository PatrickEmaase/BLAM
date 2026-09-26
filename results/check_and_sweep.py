"""
check_and_sweep.py -- two things.

(5) CEILING CHECK on real models. Reports Delta/psi_min and lambda_max, the
    two quantities that decide whether Prop. 13 and Cor. 19 can ever bite.
    Works on a single .npz.

(4) TEMPERATURE SWEEP: a zero-training-cost version of C2.

    Rescaling logits by 1/T leaves d_fg EXACTLY invariant -- d_SVD standardises
    its inputs, so any common scaling of f cancels -- while delta_m rises
    monotonically in T. So sweeping T holds the representational dissimilarity
    literally fixed and moves only the confidence floor, which is a cleaner
    causal test of Cor. 15 than label smoothing (where both move at once).

    Needs >= 2 models to form pairs. With 6 seeds you get 15 pairs at each of
    5 temperatures for the price of 6 training runs.

    python check_and_sweep.py runs/*.npz
"""

import sys, glob
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

sys.path.insert(0, '.')
import measures as ms


def load(paths):
    out = []
    for p in paths:
        z = np.load(p)
        out.append(dict(f=z['f'].astype(np.float64), g=z['g'].astype(np.float64),
                        lp=z['logp'].astype(np.float64), y=z['y'],
                        dim=int(z['dim']), seed=int(z['seed']),
                        smooth=float(z['smooth']), acc=float(z['acc']), p=p))
    return out


def geom(A, B, M):
    """Delta/psi_min and lambda_max for a pair."""
    lp, lq = A['lp'], B['lp']
    z1 = lp[:, 1:M + 1] - lp[:, [0]]
    z2 = lq[:, 1:M + 1] - lq[:, [0]]
    psi = min(z1.std(0).min(), z2.std(0).min())
    top = lp.argmax(1); n = np.arange(len(lp))
    a = lp[n, top][:, None] - lp
    off = np.ones_like(a, bool); off[n, top] = False
    Delta = float(np.where(off, a, 0).max())
    Zc = (z1 - z1.mean(0)) / np.maximum(z1.std(0), 1e-12)
    lam = float(np.linalg.eigvalsh(Zc.T @ Zc / (len(z1) - 1)).max())
    return Delta / psi, lam, psi, Delta


def retemp(m, T):
    """Logits/T, renormalised. d_fg is invariant; delta_m is not."""
    lg = m['f'] @ m['g'].T / T
    return lg - logsumexp(lg, 1, keepdims=True)


def main(paths):
    R = load(paths)
    M = R[0]['dim']
    print(f"{len(R)} model(s), M={M}\n")

    # ---------------- (5) ceiling check -------------------------------------
    print("=" * 74)
    print("(5) CEILING CHECK")
    print("=" * 74)
    if len(R) == 1:
        A = B = R[0]
        print("  only one model: self-pair, so Delta/psi and lambda_max are")
        print("  still meaningful (they are per-model geometry).\n")
    else:
        A, B = R[0], R[1]
    rat, lam, psi, Delta = geom(A, B, M)
    print(f"  psi_min                {psi:.4f}")
    print(f"  Delta                  {Delta:.4f}")
    print(f"  Delta / psi_min        {rat:.4f}")
    print(f"  lambda_max             {lam:.4f}   (<= M = {M})")
    print(f"\n  Prop. 13 ceiling:  d_fg must be < 1/(2 lambda_max) = "
          f"{1/(2*lam):.4f}")
    print(f"  Cor. 19 needs:     alpha > 4 sqrt(M) Delta/psi_min = "
          f"{4*np.sqrt(M)*rat:.1f}")
    print(f"                     against an empirical alpha*_sym ceiling ~1.35")
    print(f"  => Cor. 19 {'OUT OF REACH' if 4*np.sqrt(M)*rat > 1.35 else 'reachable'}"
          f" by a factor of {4*np.sqrt(M)*rat/1.35:.0f}x")

    if len(R) < 2:
        print("\n(4) temperature sweep needs >= 2 models. Train 6 seeds:")
        print("    for S in $(seq 0 5); do python train.py --dim 2 --seed $S"
              " --out runs; done")
        return

    # ---------------- (4) temperature sweep ---------------------------------
    print("\n" + "=" * 74)
    print("(4) TEMPERATURE SWEEP  (d_fg held EXACTLY fixed)")
    print("=" * 74)
    base = [m for m in R if m['smooth'] == 0.0]
    pairs = [(base[i], base[j]) for i in range(len(base))
             for j in range(i + 1, len(base))]
    print(f"  {len(base)} models -> {len(pairs)} pairs")

    d_ref = [ms.d_fg(a['f'], a['g'], b['f'], b['g'], a['y'], M)
             for a, b in pairs]
    print(f"\n  {'T':>6} {'delta_3':>11} {'mean KL':>9} {'mean d_fg':>10} "
          f"{'spearman':>9} {'p':>8}")
    for T in [1.0, 2.0, 4.0, 8.0, 16.0]:
        kl, dd, dm = [], [], []
        for (a, b), dr in zip(pairs, d_ref):
            la, lb = retemp(a, T), retemp(b, T)
            kl.append(ms.d_kl(la, lb))
            dd.append(dr)                      # invariant by construction
            ps = np.sort(np.exp(la), 1)[:, ::-1]
            dm.append(ps[:, :M + 1].min())
        rho, p = spearmanr(kl, dd)
        print(f"  {T:>6.1f} {np.median(dm):>11.3e} {np.mean(kl):>9.4f} "
              f"{np.mean(dd):>10.4f} {rho:>9.3f} {p:>8.4f}")

    # invariance assertion -- this is the whole point of the design
    T = 8.0
    d_hot = [ms.d_fg(a['f'] / T, a['g'], b['f'] / T, b['g'], a['y'], M)
             for a, b in pairs]
    err = max(abs(x - y) for x, y in zip(d_ref, d_hot))
    print(f"\n  d_fg invariance under rescaling: max |change| = {err:.2e}"
          f"  {'OK' if err < 1e-6 else 'FAIL'}")
    print("  (d_fg is fixed by construction, so any change in the rank")
    print("   correlation is attributable to delta_m alone)")


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob('runs/*.npz'))
    if not args:
        print(__doc__); sys.exit(1)
    main(args)
