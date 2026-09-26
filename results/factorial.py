"""
(0) CORRECTNESS FIX.  alpha* was computed only over y with b_y > a_y, i.e. the
    forward direction D_alpha(p||p'). Renyi is asymmetric. If b_y <= a_y for
    every y then the forward divergence decays for all finite alpha
    (alpha* = inf) while the REVERSE direction diverges. A construction is only
    a counterexample if BOTH directions stay small, so the meaningful quantity
    is
        alpha*_sym = min( alpha*(p||p'), alpha*(p'||p) ).
    Every dihedral cell reporting alpha* = inf must be rechecked this way.

(1) Factorial design isolating the shrink confound.
(2) Targeted search over reflection-type and reflection-composed permutations.
"""

import numpy as np
from itertools import permutations
from harness import make_pair, gaps_and_margin, dfg

MF = 0.08


def astar_dir(A, B):
    m = B > A + 1e-8
    return float(np.min(B[m] / (B[m] - A[m]))) if m.any() else np.inf


def astar_sym(a, b, off, margin, ok, m=MF):
    sel = ok & (margin >= m)
    if sel.sum() < 5:
        return np.nan, np.nan, np.nan, 0
    A = a[sel][off[sel]]; B = b[sel][off[sel]]
    f = astar_dir(A, B)          # forward: p || p'
    r = astar_dir(B, A)          # reverse: p' || p
    return min(f, r), f, r, int(sel.sum())


def is_dihedral(s, k):
    s = np.asarray(s)
    return any(np.all(s == (a * np.arange(k) + b) % k)
               for a in (1, -1) for b in range(k))


def is_reflection(s, k):
    s = np.asarray(s)
    return any(np.all(s == (-np.arange(k) + b) % k) for b in range(k))


def run(M, k, sigma, nu, irregular, shrink, seed=11, n=2400):
    rng = np.random.default_rng(seed)
    F, G, Fp, Gp, lab, sg, U, lam = make_pair(
        M, k, n, rng, sigma=sigma, irregular=irregular,
        nonuniform=nu, shrink=shrink)
    a, b, off, margin, ok = gaps_and_margin(F, G, Fp, Gp)
    if ok.mean() < 0.6:
        return None
    As, af, ar, ns = astar_sym(a, b, off, margin, ok)
    d = dfg(F, G, Fp, Gp, lab, M)
    if ns < 30 or not np.isfinite(d) or np.isnan(As):
        return None
    return As, af, ar, d, ok.mean()


if __name__ == "__main__":
    # =============================================== (0) recheck the dihedral cells
    print("=" * 80)
    print("(0) SYMMETRIC alpha* ON THE DIHEDRAL CELLS (k=6, M=2, all 720 perms)")
    print("=" * 80)
    k = 6
    for nu in [False, True]:
        rot = dict(s=[], f=[], r=[], d=[])
        ref = dict(s=[], f=[], r=[], d=[])
        brk = dict(s=[], f=[], r=[], d=[])
        for s in permutations(range(k)):
            out = run(2, k, list(s), nu, False, 0.0)
            if out is None:
                continue
            As, af, ar, d, _ = out
            g = (ref if is_reflection(s, k) else rot) if is_dihedral(s, k) else brk
            g['s'].append(min(As, 50)); g['f'].append(min(af, 50))
            g['r'].append(min(ar, 50)); g['d'].append(d)
        print(f"\nnorms {'non-uniform' if nu else 'uniform'}")
        print(f"  {'class':<18} {'n':>4} {'med a*_fwd':>11} {'med a*_rev':>11} "
              f"{'med a*_sym':>11} {'max d_fg':>9} {'both-large':>11}")
        for nm, g in [("rotation", rot), ("reflection", ref), ("order-breaking", brk)]:
            if not g['s']:
                continue
            S = np.array(g['s']); Fw = np.array(g['f'])
            Rv = np.array(g['r']); D = np.array(g['d'])
            bl = int(((S > 1.3) & (D > 0.15)).sum())
            print(f"  {nm:<18} {len(S):>4} {np.median(Fw):>11.3f} "
                  f"{np.median(Rv):>11.3f} {np.median(S):>11.3f} "
                  f"{D.max():>9.3f} {bl:>11d}")

    # ======================================================== (1) factorial design
    print("\n" + "=" * 80)
    print("(1) FACTORIAL: shrink x symmetry x norms x M, blocked on permutation class")
    print("=" * 80)
    print(f"{'M':>2} {'shrink':>7} {'irreg':>6} {'nonunif':>8} {'class':>8} "
          f"{'acc':>5} {'med a*_sym':>11} {'max a*_sym':>11} {'med d_fg':>9}")

    k = 8
    rng0 = np.random.default_rng(3)
    pool_brk = [rng0.permutation(k) for _ in range(40)]
    pool_brk = [s for s in pool_brk if not is_dihedral(s, k)][:24]
    pool_dih = [(-np.arange(k) + b) % k for b in range(4)] + \
               [(np.arange(k) + b) % k for b in range(1, 4)]

    cells = {}
    for M in [2, 3]:
        for shrink in [0.0, 0.25, 0.5]:
            for irreg in [False, True]:
                for nu in [False, True]:
                    for cname, pool in [("dihedral", pool_dih), ("breaking", pool_brk)]:
                        S, D, acc = [], [], 0
                        for s in pool:
                            out = run(M, k, list(s), nu, irreg, shrink)
                            if out is None:
                                continue
                            acc += 1
                            S.append(min(out[0], 50)); D.append(out[3])
                        if not S:
                            continue
                        S, D = np.array(S), np.array(D)
                        cells[(M, shrink, irreg, nu, cname)] = (S, D)
                        print(f"{M:>2} {shrink:>7.2f} {str(irreg):>6} {str(nu):>8} "
                              f"{cname:>8} {acc:>5} {np.median(S):>11.3f} "
                              f"{S.max():>11.3f} {np.median(D):>9.3f}")

    # marginal effects on max alpha*_sym
    print("\nMARGINAL EFFECTS (max alpha*_sym over cells sharing a level)")
    def marg(idx, name):
        lv = sorted({key[idx] for key in cells})
        out = []
        for v in lv:
            m = max(cells[key][0].max() for key in cells if key[idx] == v)
            out.append(f"{v}={m:.3f}")
        print(f"  {name:<10} " + "   ".join(out))
    marg(0, "M"); marg(1, "shrink"); marg(2, "irregular"); marg(3, "nonuniform")
    marg(4, "perm class")

    # ============================== (2) reflection / reflection-composed search ---
    print("\n" + "=" * 80)
    print("(2) REFLECTION-TARGETED SEARCH (M=2)")
    print("=" * 80)
    best = None
    rows = []
    for k in [6, 8, 10, 12]:
        refl = [(-np.arange(k) + b) % k for b in range(k)]
        comp = []
        rg = np.random.default_rng(7)
        for b in range(k):                       # reflection composed with a swap
            for _ in range(6):
                s = ((-np.arange(k) + b) % k).copy()
                i, j = rg.choice(k, 2, replace=False)
                s[[i, j]] = s[[j, i]]
                comp.append(s)
        for nm, pool in [("reflection", refl), ("refl+swap", comp)]:
            for irreg in [False, True]:
                for nu in [False, True]:
                    for s in pool:
                        out = run(2, k, list(s), nu, irreg, 0.25)
                        if out is None:
                            continue
                        As, af, ar, d, _ = out
                        rows.append((k, nm, irreg, nu, min(As, 50), d))
                        if d > 0.15 and (best is None or As > best[4]):
                            best = (k, nm, irreg, nu, As, d)

    R = [r for r in rows if r[5] > 0.15]
    print(f"configs with d_fg > 0.15: {len(R)} / {len(rows)}")
    if R:
        top = sorted(R, key=lambda r: -r[4])[:8]
        print(f"{'k':>3} {'family':>11} {'irreg':>6} {'nonunif':>8} "
              f"{'a*_sym':>8} {'d_fg':>7}")
        for r in top:
            print(f"{r[0]:>3} {r[1]:>11} {str(r[2]):>6} {str(r[3]):>8} "
                  f"{r[4]:>8.3f} {r[5]:>7.3f}")
    print(f"\nbest (a*_sym, d_fg>0.15): {best}")
