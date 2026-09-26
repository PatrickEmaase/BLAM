"""
analyze.py -- experiments C1-C4 from the .npz files train.py writes.

    python analyze.py --runs runs/ --out results/

C1  delta_m profile against m, and divergences restricted to each input's
    top-k versus its tail. Decides whether Remark 16's pessimism about
    delta_m holds on real models.
C2  label-smoothing intervention: does raising delta_m restore KL's ability
    to predict representational dissimilarity?
C3  non-vacuity fraction of Thm 12 per representation dimension.
C4  robustness across d_fg, mCCA, linear CKA and Procrustes.

Prints a report and writes results/*.csv. Also runs the CI guards: any pair
dropped by a filter is counted and reported, never silently discarded.
"""

import argparse, glob, os, itertools, json
import numpy as np
from scipy.stats import spearmanr
import measures as ms


def load(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, '*.npz'))):
        z = np.load(p)
        out.append(dict(f=z['f'].astype(np.float64), g=z['g'].astype(np.float64),
                        lp=z['logp'].astype(np.float64), y=z['y'],
                        acc=float(z['acc']), loss=float(z['loss']),
                        dim=int(z['dim']), seed=int(z['seed']),
                        smooth=float(z['smooth']),
                        width=int(z['width']) if 'width' in z else 0,
                        nclass=int(z['nclass']) if 'nclass' in z else 0,
                        path=p))
    return out


def pair_row(A, B, M):
    lp, lq = A['lp'], B['lp']
    r = dict(dim=M, sA=A['seed'], sB=B['seed'], smooth=A['smooth'],
             width=A['width'], nclass=A['nclass'],
             loss=0.5 * (A['loss'] + B['loss']),
             accA=A['acc'], accB=B['acc'])
    r['kl'] = ms.d_kl(lp, lq)
    r['kl_rev'] = ms.d_kl(lq, lp)
    r['tv'] = ms.d_tv(lp, lq)
    r['h2'] = ms.d_hellinger2(lp, lq)
    r['js'] = ms.d_js(lp, lq)
    for al in [0.5, 1.5, 2.0, 4.0, np.inf]:
        r[f'renyi_{al}'] = max(ms.d_renyi(lp, lq, al), ms.d_renyi(lq, lp, al))
    r['dfg'] = ms.d_fg(A['f'], A['g'], B['f'], B['g'], A['y'], M)
    r['mcca'] = ms.m_cca(A['f'], B['f'])
    r['cka'] = ms.linear_cka(A['f'], B['f'])
    r['proc'] = ms.procrustes(A['f'], B['f'])
    ast, nsel, okfrac = ms.alpha_star_sym(lp, lq)
    r['astar_sym'], r['n_sel'], r['argmax_agree'] = ast, nsel, okfrac
    b, rho2, psi, lm = ms.thm12_bound(lp, lq, M)
    r['bound'], r['rho2'], r['psi'], r['lam_max'] = b, rho2, psi, lm
    dp = ms.delta_profile(lp, lq)
    for m in dp:
        r[f'delta_{m}'] = dp[m]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = load(a.runs)
    print(f"loaded {len(R)} models from {a.runs}")
    if not R:
        raise SystemExit("no runs found")

    rows, dropped = [], 0
    cells = sorted({(m['dim'], m['nclass'], m['width'], m['smooth'])
                    for m in R})
    for (M, nc, w, sm) in cells:
        if True:
            grp = [m for m in R if (m['dim'], m['nclass'], m['width'],
                                    m['smooth']) == (M, nc, w, sm)]
            for A, B in itertools.combinations(grp, 2):
                try:
                    rows.append(pair_row(A, B, M))
                except Exception as e:
                    dropped += 1
                    print(f"  DROPPED pair ({A['seed']},{B['seed']}): {e}")
    print(f"pairs: {len(rows)}   dropped: {dropped}"
          f"   (dropped must be 0 or explained)")

    keys = sorted(rows[0].keys())
    with open(os.path.join(a.out, 'pairs.csv'), 'w') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(str(r.get(k, '')) for k in keys) + '\n')

    def sub(**kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    # ------------------------------------------------------------- C1 ------
    print("\n" + "=" * 70 + "\nC1  delta_m profile and tail localisation\n" + "=" * 70)
    base = sub(smooth=0.0)
    for M in sorted({r['dim'] for r in base}):
        g = [r for r in base if r['dim'] == M]
        print(f"\n dim {M}  (n={len(g)} pairs)")
        print(f"   {'m':>3} {'median delta_m':>16} {'1/delta_m':>12}")
        for m in [2, 3, 4, 5, 10]:
            k = f'delta_{m}'
            if k in g[0]:
                v = np.median([r[k] for r in g])
                print(f"   {m:>3} {v:>16.3e} {1/max(v,1e-300):>12.3e}")
    print("\n  top-k vs tail: KL and d_fg on restricted supports")
    print(f"   {'k':>3} {'KL(top-k)':>11} {'KL(tail)':>11} {'frac of full KL':>17}")
    R0 = [m for m in R if m['smooth'] == 0.0]
    for M in sorted({m['dim'] for m in R0}):
        grp = [m for m in R0 if m['dim'] == M][:6]
        for k in [1, 2, 3, 5]:
            kt, kb, kf = [], [], []
            for A, B in itertools.combinations(grp, 2):
                full = ms.d_kl(A['lp'], B['lp'])
                pt, qt = ms.topk_pair(A['lp'], B['lp'], k)
                pb, qb = ms.tail_pair(A['lp'], B['lp'], k)
                kt.append(ms.d_kl(pt, qt))
                kb.append(ms.d_kl(pb, qb))
                kf.append(kt[-1] / max(full, 1e-12))
            print(f"   {k:>3} {np.median(kt):>11.4f} {np.median(kb):>11.4f} "
                  f"{np.median(kf):>17.3f}   (dim {M})")

    # -------------------------------------------- 7.2a width sweep ---------
    widths = sorted({r['width'] for r in rows if r['width'] > 0})
    if len(widths) > 1:
        import numpy.linalg as la
        print("\n" + "=" * 70 + "\n7.2a  width sweep, matched-loss control\n"
              + "=" * 70)
        print(f"{'c':>3} {'w':>5} {'n':>4} {'mean loss':>10} {'mean KL':>9} "
              f"{'mean d_fg':>10} {'sd d_fg':>9}")
        for nc in sorted({r['nclass'] for r in rows if r['width'] > 0}):
            for w in widths:
                g = [r for r in rows if r['nclass'] == nc and r['width'] == w
                     and r['smooth'] == 0.0]
                if not g:
                    continue
                d = np.array([r['dfg'] for r in g])
                print(f"{nc:>3} {w:>5} {len(g):>4} "
                      f"{np.mean([r['loss'] for r in g]):>10.4f} "
                      f"{np.mean([r['kl'] for r in g]):>9.4f} "
                      f"{d.mean():>10.4f} {d.std():>9.4f}")
        print("\n  regression of d_fg on log(width), with loss as covariate")
        for nc in sorted({r['nclass'] for r in rows if r['width'] > 0}):
            g = [r for r in rows if r['nclass'] == nc and r['width'] > 0
                 and r['smooth'] == 0.0]
            if len(g) < 6:
                continue
            yv = np.array([r['dfg'] for r in g])
            lw = np.log([r['width'] for r in g])
            ls = np.array([r['loss'] for r in g])
            bn = la.lstsq(np.column_stack([np.ones(len(g)), lw]), yv,
                          rcond=None)[0]
            ba = la.lstsq(np.column_stack([np.ones(len(g)), lw, ls]), yv,
                          rcond=None)[0]
            print(f"   c={nc}: naive {bn[1]:+.4f}   loss-adjusted {ba[1]:+.4f}"
                  f"   loss coef {ba[2]:+.4f}")
        print("  (a negative loss-adjusted slope is the claim that wider"
              " networks learn more similar representations)")

    # ------------------------------------------------------------- C2 ------
    print("\n" + "=" * 70 + "\nC2  label smoothing: does raising delta_m restore KL?\n" + "=" * 70)
    print(f"{'dim':>4} {'smooth':>7} {'n':>4} {'median delta_3':>15} "
          f"{'mean KL':>9} {'mean d_fg':>10} {'spearman':>9} {'p':>8}")
    for M in sorted({r['dim'] for r in rows}):
        for sm in sorted({r['smooth'] for r in rows if r['dim'] == M}):
            g = sub(dim=M, smooth=sm)
            if len(g) < 4:
                continue
            kl = np.array([r['kl'] for r in g]); d = np.array([r['dfg'] for r in g])
            rho, p = spearmanr(kl, d)
            d3 = np.median([r.get('delta_3', np.nan) for r in g])
            print(f"{M:>4} {sm:>7} {len(g):>4} {d3:>15.3e} {kl.mean():>9.4f} "
                  f"{d.mean():>10.4f} {rho:>9.3f} {p:>8.4f}")

    # ------------------------------------------------------------- C3 ------
    print("\n" + "=" * 70 + "\nC3  non-vacuity of Thm 12\n" + "=" * 70)
    print(f"{'dim':>4} {'n':>5} {'violations':>11} {'non-vacuous':>12} "
          f"{'median bound':>13} {'median d_fg':>12}")
    for M in sorted({r['dim'] for r in rows}):
        g = sub(dim=M, smooth=0.0)
        b = np.array([r['bound'] for r in g]); d = np.array([r['dfg'] for r in g])
        print(f"{M:>4} {len(g):>5} {int((d > b + 1e-9).sum()):>11} "
              f"{(b < 1).mean():>11.1%} {np.median(b):>13.3f} "
              f"{np.median(d):>12.3f}")

    # ------------------------------------------------------------- C4 ------
    print("\n" + "=" * 70 + "\nC4  agreement across similarity measures\n" + "=" * 70)
    g = sub(smooth=0.0)
    kl = np.array([r['kl'] for r in g])
    print(f"{'measure':>10} {'spearman with KL':>18} {'p':>9}")
    for k, nm in [('dfg', 'd_fg'), ('mcca', 'mCCA'), ('cka', 'CKA'),
                  ('proc', 'Procrustes')]:
        v = np.array([r[k] for r in g])
        rho, p = spearmanr(kl, v)
        print(f"{nm:>10} {rho:>18.3f} {p:>9.4f}")

    print("\nCI guards")
    am = np.array([r['argmax_agree'] for r in rows])
    print(f"  argmax agreement: min {am.min():.3f}, median {np.median(am):.3f}")
    inf_pairs = [r for r in rows if not np.isfinite(r['astar_sym'])]
    if inf_pairs:
        n = len(R[0]['lp'])
        bad = [r for r in inf_pairs if abs(r['dfg']) > 5.0 / n]
        print(f"  alpha*_sym = inf on {len(inf_pairs)} pairs; "
              f"{len(bad)} violate d_fg < 5/n  (must be 0)")
    else:
        print("  alpha*_sym finite on all pairs (expected: no pair is ~_L equivalent)")
    print(f"\nwrote {os.path.join(a.out,'pairs.csv')}")


if __name__ == '__main__':
    main()
