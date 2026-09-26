"""
Corrected rate lemma + verification.

Leading-order expansion (per input x, c=1):

  log S_alpha = log sum_y p^alpha q^(1-alpha)
              ~= sum_{y != yhat} [ e^{-rho E_a(y)} - alpha e^{-rho a_y}
                                                  + (alpha-1) e^{-rho b_y} ]

Group the three exponent families {E_a(y)}, {a_y}, {b_y} by VALUE. At value v
the total coefficient is

  C(v) = #{y: E_a(y)=v} - alpha*#{y: a_y=v} + (alpha-1)*#{y: b_y=v}.

The decay rate is the smallest v > 0 with C(v) != 0. Labels with a_y = b_y give
E_a(y) = a_y = b_y and contribute 1 - alpha + (alpha-1) = 0: they cancel
exactly and must be EXCLUDED. This is the correction to the naive
kappa = min{E, a, b}.

At alpha = 1 every first-order coefficient vanishes, so KL is governed by the
second-order term  rho * sum (b_y - a_y) e^{-rho a_y}, giving

  D_KL = Theta( rho * e^{-rho * a_min'} ),   a_min' = min{a_y : b_y != a_y}.

Above alpha*, D_alpha = Theta(rho): the integrand diverges like
e^{rho|E_a|} but Renyi takes a log, so growth is LINEAR with slope
|E_a,min| / (alpha - 1), tending to max_y (b_y - a_y) as alpha -> inf.
"""

import numpy as np
from ladder import build, gaps, logp, renyi, d_fg, llv_t1, tv, hellinger2, jsd

TOL = 1e-9


def rate_prediction(aa, bb, alpha):
    """Exponential decay rate of D_alpha, with cancellation handled."""
    keep = np.abs(aa - bb) > TOL          # drop exactly-cancelling labels
    a, b = aa[keep], bb[keep]
    if len(a) == 0:
        return np.inf
    E = b + alpha * (a - b)
    if abs(alpha - 1.0) < TOL:
        return a.min()                    # second-order; prefactor rho
    vals = np.concatenate([E, a, b])
    coef = np.concatenate([np.ones_like(E),
                           -alpha * np.ones_like(a),
                           (alpha - 1) * np.ones_like(b)])
    order = np.argsort(vals)
    vals, coef = vals[order], coef[order]
    i = 0
    while i < len(vals):
        j = i
        tot = 0.0
        while j < len(vals) and vals[j] - vals[i] < 1e-7:
            tot += coef[j]
            j += 1
        if abs(tot) > 1e-8 and vals[i] > TOL:
            return vals[i]
        i = j
    return np.inf


def alpha_star(aa, bb):
    m = bb > aa + TOL
    return np.min(bb[m] / (bb[m] - aa[m])) if m.any() else np.inf


def growth_slope(aa, bb, alpha):
    """Predicted linear growth slope of D_alpha above alpha*."""
    keep = np.abs(aa - bb) > TOL
    a, b = aa[keep], bb[keep]
    if np.isinf(alpha):
        return (b - a).max()
    E = b + alpha * (a - b)
    return max(0.0, -E.min()) / (alpha - 1)


def analyse(k=7, n_per=120, seed=0, perm_seed=1,
            rhos=np.array([6, 9, 12, 15, 18, 21, 24, 27, 30], float), verbose=True):
    rng = np.random.default_rng(perm_seed)
    sigma = rng.permutation(k)
    while np.all(sigma == np.arange(k)):
        sigma = rng.permutation(k)

    aa_, bb_, lab = gaps(k, n_per, seed, sigma)
    off = np.ones_like(aa_, bool)
    off[np.arange(len(lab)), lab] = False
    aa, bb = aa_[off], bb_[off]

    ast = alpha_star(aa, bb)
    alphas = [0.5, 0.9, 1.0, 1.01, 1.02, 1.03, 1.05, 1.08, 1.2, 1.5,
              2.0, 3.0, 5.0, 10.0, 30.0, np.inf]

    D = {al: np.array([renyi(*_lp(k, n_per, r, seed, sigma), al) for r in rhos])
         for al in alphas}

    if verbose:
        print(f"k={k} sigma={sigma.tolist()}   predicted alpha* = {ast:.4f}")
        ncancel = int(np.sum(np.abs(aa - bb) <= TOL))
        print(f"labels with a_y == b_y (exactly cancelling): "
              f"{ncancel}/{len(aa)}")
        print(f"a_min(all)={aa.min():.4f}  a_min(non-cancel)="
              f"{aa[np.abs(aa-bb)>TOL].min():.4f}\n")
        print(f"{'alpha':>7} {'branch':>7} {'pred':>10} {'fit':>10} "
              f"{'rel.err':>9}")

    out = []
    for al in alphas:
        y = D[al]
        growing = y[-1] > y[0]
        if growing:
            pred = growth_slope(aa, bb, al)
            fit = np.polyfit(rhos, y, 1)[0]              # linear in rho
            branch = "linear"
        else:
            pred = rate_prediction(aa, bb, al)
            adj = y / rhos if abs(al - 1.0) < TOL else y  # KL prefactor
            fit = -np.polyfit(rhos, np.log(np.abs(adj)), 1)[0]
            branch = "exp"
        err = abs(fit - pred) / max(abs(pred), 1e-12)
        out.append((al, branch, pred, fit, err))
        if verbose:
            nm = "inf" if np.isinf(al) else f"{al:g}"
            print(f"{nm:>7} {branch:>7} {pred:>10.4f} {fit:>10.4f} {err:>9.2%}")

    grid_cross = min([al for al in alphas if D[al][-1] > D[al][0]], default=None)
    if verbose:
        print(f"\nmeasured crossover <= {grid_cross}   predicted alpha* = {ast:.4f}")
    return ast, grid_cross, out


def _lp(k, n_per, rho, seed, sigma):
    f1, g1, f2, g2, *_ = build(k, n_per, rho, seed, sigma)
    return logp(f1, g1), logp(f2, g2)


def scatter(n_perm=25, k=7, n_per=80, seed=0):
    """A3 preview: predicted alpha* vs measured crossover, many permutations."""
    fine = np.concatenate([np.linspace(1.001, 1.5, 60), np.linspace(1.5, 8, 60)])
    rows = []
    for ps in range(1, n_perm + 1):
        rng = np.random.default_rng(ps)
        sigma = rng.permutation(k)
        if np.all(sigma == np.arange(k)):
            continue
        aa_, bb_, lab = gaps(k, n_per, seed, sigma)
        off = np.ones_like(aa_, bool)
        off[np.arange(len(lab)), lab] = False
        aa, bb = aa_[off], bb_[off]
        ast = alpha_star(aa, bb)
        lo = _lp(k, n_per, 10.0, seed, sigma)
        hi = _lp(k, n_per, 26.0, seed, sigma)
        meas = None
        for al in fine:
            if renyi(*hi, al) > renyi(*lo, al):
                meas = al
                break
        if meas is not None and np.isfinite(ast):
            rows.append((ast, meas))
    A = np.array(rows)
    r = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
    med = np.median(np.abs(A[:, 1] - A[:, 0]) / A[:, 0])
    print(f"\nA3: {len(A)} permutations   Pearson r = {r:.4f}   "
          f"median |meas-pred|/pred = {med:.2%}")
    print(f"    alpha* range: [{A[:,0].min():.3f}, {A[:,0].max():.3f}]")
    return A


if __name__ == "__main__":
    analyse()
    scatter()
