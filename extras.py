"""Computes what the old analyze.py omits, straight from pairs.csv.
   python extras.py results/cifar/pairs.csv [--min-acc 0.9]"""
import sys, csv
import numpy as np
from scipy.stats import spearmanr

path = sys.argv[1]
mina = float(sys.argv[sys.argv.index('--min-acc') + 1]) if '--min-acc' in sys.argv else 0.0

rows = []
with open(path) as fh:
    for r in csv.DictReader(fh):
        d = {}
        for k, v in r.items():
            try: d[k] = float(v)
            except (ValueError, TypeError): d[k] = v
        rows.append(d)
n0 = len(rows)
if mina > 0:
    rows = [r for r in rows if min(r.get('accA', 1), r.get('accB', 1)) >= mina]
print(f"{path}: {len(rows)}/{n0} pairs" + (f"  (acc >= {mina})" if mina else ""))

def boot(g, stat, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    sd = sorted({r['sA'] for r in g} | {r['sB'] for r in g})
    idx = {s: i for i, s in enumerate(sd)}
    out = []
    for _ in range(B):
        take = set(rng.choice(len(sd), len(sd), replace=True).tolist())
        sel = [r for r in g if idx[r['sA']] in take and idx[r['sB']] in take]
        if len(sel) >= 3:
            try: out.append(stat(sel))
            except Exception: pass
    return (np.nan, np.nan) if len(out) < 50 else tuple(np.percentile(out, [2.5, 97.5]))

sp = lambda S, a, b: spearmanr([r[a] for r in S], [r[b] for r in S])[0]

print("\n=== Spearman(KL, d_fg) with 95% cluster bootstrap CI over seeds ===")
print(f"{'dim':>4} {'smooth':>7} {'n':>5} {'rho':>8} {'95% CI':>18} {'p':>9}")
for key in sorted({(r['dim'], r['smooth']) for r in rows}):
    g = [r for r in rows if (r['dim'], r['smooth']) == key]
    if len(g) < 4: continue
    rho, p = spearmanr([r['kl'] for r in g], [r['dfg'] for r in g])
    lo, hi = boot(g, lambda S: sp(S, 'kl', 'dfg'))
    print(f"{key[0]:>4.0f} {key[1]:>7.2f} {len(g):>5} {rho:>8.3f} "
          f"  [{lo:>6.3f}, {hi:>6.3f}] {p:>9.4f}")

print("\n=== C4 per dimension (pooling mixes dimensions) ===")
print(f"{'dim':>4} {'n':>5} {'d_fg':>9} {'mCCA':>9} {'CKA':>9} {'Procrustes':>11}")
for M in sorted({r['dim'] for r in rows}):
    g = [r for r in rows if r['dim'] == M and r['smooth'] == 0.0]
    if len(g) < 4: continue
    v = [sp(g, 'kl', k) for k in ('dfg', 'mcca', 'cka', 'proc')]
    print(f"{M:>4.0f} {len(g):>5} " + " ".join(f"{x:>9.3f}" for x in v[:3])
          + f" {v[3]:>11.3f}")

if any(r.get('width', 0) for r in rows):
    print("\n=== width sweep (after filter) ===")
    print(f"{'c':>3} {'w':>5} {'n':>5} {'mean d_fg':>10} {'mean mCCA':>10}")
    for c in sorted({r['nclass'] for r in rows if r['width']}):
        for w in sorted({r['width'] for r in rows if r['nclass'] == c}):
            g = [r for r in rows if r['nclass'] == c and r['width'] == w
                 and r['smooth'] == 0.0]
            if len(g) < 3: continue
            print(f"{c:>3.0f} {w:>5.0f} {len(g):>5} "
                  f"{np.mean([r['dfg'] for r in g]):>10.4f} "
                  f"{np.mean([r['mcca'] for r in g]):>10.4f}")
