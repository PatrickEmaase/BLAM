"""
A3, rebuilt.

The old panel (b) plotted predicted vs measured alpha* and spanned only
[1.01, 1.24] -- no dynamic range, reads as "the threshold is just 1".

Replaced by two panels that do have range and that test the parts of the rate
lemma which are actually non-obvious:
  (b) predicted vs measured DECAY RATE kappa(alpha), which exercises the
      cancellation rule and spans ~3 orders of magnitude;
  (c) alpha* as a function of the MARGIN FLOOR of the input distribution,
      making explicit that the threshold is set by the worst-margin input.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from perx import one_x, D_alpha_exact, astar, rate_pred
from tradeoff import all_gaps

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False})

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.15))

# ------------------------------------------------------- (a) ladder, as before
k, j = 7, 0
sigma = np.random.default_rng(1).permutation(k)
phi = 0.25 * np.pi / k
a, b, af, bf = one_x(k, phi, j, sigma)
ast = astar(a, b)
rhos = np.linspace(5, 120, 60)

for al, c in [(0.5, "#1b6ca8"), (1.0, "#d1495b"), (1.2, "#edae49"),
              (2.0, "#2e4057"), (np.inf, "#8d5a97")]:
    y = [D_alpha_exact(af, bf, r, al) for r in rhos]
    lab = (r"$\alpha=\infty$" if np.isinf(al) else
           (r"$\alpha=1$ (KL)" if al == 1.0 else rf"$\alpha={al:g}$"))
    ax[0].plot(rhos, np.maximum(y, 1e-12), color=c, lw=1.2, label=lab)
ax[0].axvline(0, color="none")
ax[0].set_yscale("log")
ax[0].set_xlabel(r"unembedding norm $\rho$")
ax[0].set_ylabel(r"$D_\alpha$")
ax[0].set_title(r"(a) divergence ladder", fontsize=8)
ax[0].legend(fontsize=6, loc="lower left")
for val, c in [(0.6326, "#d1495b"), (0.3780, "#1b6ca8"), (1.4220, "#2e4057")]:
    ax[0].plot(rhos, np.full_like(rhos, val), lw=0.8, color=c, alpha=0.3)

# ------------------------------------ (b) predicted vs measured decay rate ---
rho_fit = np.array([60, 80, 100, 120, 140, 160], float)
P, Mz = [], []
for ps in range(1, 26):
    s = np.random.default_rng(ps).permutation(k)
    if np.all(s == np.arange(k)):
        continue
    for ph in np.linspace(-0.7, 0.7, 5) * np.pi / k:
        aa, bb, A, B = one_x(k, ph, 0, s)
        t = astar(aa, bb)
        if not np.isfinite(t):
            continue
        for frac in [0.35, 0.6, 0.8, 0.9, 0.95, 0.98]:
            al = 1.0 + (t - 1.0) * frac
            kp = rate_pred(aa, bb, al)
            if not np.isfinite(kp) or kp <= 1e-6:
                continue
            y = np.array([D_alpha_exact(A, B, r, al) for r in rho_fit])
            if np.any(y <= 0) or y[-1] > y[0]:
                continue
            kf = -np.polyfit(rho_fit, np.log(y), 1)[0]
            if kf > 0:
                P.append(kp); Mz.append(kf)

P, Mz = np.array(P), np.array(Mz)
ax[1].scatter(P, Mz, s=5, color="#1b6ca8", alpha=0.45, linewidths=0)
lim = [min(P.min(), Mz.min()) * 0.6, max(P.max(), Mz.max()) * 1.6]
ax[1].plot(lim, lim, "k--", lw=0.7)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel(r"predicted rate $\kappa(\alpha)$")
ax[1].set_ylabel(r"measured rate")
med = np.median(np.abs(Mz - P) / P)
dec = np.log10(P.max() / P.min())
ax[1].set_title(rf"(b) decay rate, med.\ err {med:.1%}", fontsize=8)

# ------------------------- (c) margin-conditioned alpha*_m, several M --------
import numpy as _np
curves = _np.load("panelc.npy", allow_pickle=True)
for MM, A, c, ls, lb in curves:
    ax[2].plot(MM, A, ls, color=c, lw=1.1, label=lb)
ax[2].set_xlabel(r"margin floor $m$ (frac.\ of max margin)")
ax[2].set_ylabel(r"$\alpha^\star_m$")
ax[2].axhline(1.0, color="0.6", lw=0.6, ls=":")
ax[2].set_title(r"(c) $\alpha^\star_m$ vs margin floor", fontsize=8)
ax[2].legend(fontsize=6, loc="upper left")

plt.tight_layout()
plt.savefig("fig_ladder.pdf", bbox_inches="tight")
print(f"panel b: n={len(P)}, median rel err = {med:.3%}, "
      f"dynamic range = {dec:.2f} decades")
print(f"panel b: kappa in [{P.min():.2e}, {P.max():.2e}]")
