import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from perx import one_x, D_alpha_exact, astar

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False})

k, j = 7, 0
sigma = np.random.default_rng(1).permutation(k)
phi = 0.25 * np.pi / k
a, b, af, bf = one_x(k, phi, j, sigma)
ast = astar(a, b)

# ---------------------------------------------------------------- Figure 1 --
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.3))
rhos = np.linspace(5, 120, 60)

for al, c, ls in [(0.5, "#1b6ca8", "-"), (1.0, "#d1495b", "-"),
                  (1.2, "#edae49", "-"), (1.34, "#66a182", "--"),
                  (2.0, "#2e4057", "-"), (np.inf, "#8d5a97", "-")]:
    y = [D_alpha_exact(af, bf, r, al) for r in rhos]
    lab = r"$\alpha=\infty$" if np.isinf(al) else (
        r"$\alpha=1$ (KL)" if al == 1.0 else
        (r"$\alpha\approx\alpha^\star$" if al == 1.34 else rf"$\alpha={al:g}$"))
    ax[0].plot(rhos, np.maximum(y, 1e-12), ls, color=c, lw=1.2, label=lab)

ax[0].set_yscale("log")
ax[0].set_xlabel(r"unembedding norm $\rho$")
ax[0].set_ylabel(r"$D_\alpha$")
ax[0].set_title(r"(a) divergence ladder", fontsize=8)
ax[0].legend(fontsize=6, ncol=2, loc="lower left")
ax[0].axhline(1.0, color="0.7", lw=0.5, ls=":")

# flat representational measures
for val, nm, c in [(0.6326, r"$d_{\mathbf{f},\mathbf{g}}$", "#d1495b"),
                   (0.3780, r"$m_{\mathrm{CCA}}$", "#1b6ca8"),
                   (1.4220, r"$d^\lambda_{\mathrm{LLV}}$", "#2e4057")]:
    ax[0].plot(rhos, np.full_like(rhos, val), lw=0.8, color=c, alpha=0.35)

# panel b: predicted vs measured crossover across geometries
pred, meas = [], []
for ps in range(1, 60):
    s = np.random.default_rng(ps).permutation(k)
    if np.all(s == np.arange(k)):
        continue
    for p in np.linspace(-0.7, 0.7, 5) * np.pi / k:
        aa, bb, A, B = one_x(k, p, 0, s)
        t = astar(aa, bb)
        if not np.isfinite(t):
            continue
        lo, hi = 1.0, 30.0
        for _ in range(45):
            m = 0.5 * (lo + hi)
            if D_alpha_exact(A, B, 200, m) > D_alpha_exact(A, B, 60, m):
                hi = m
            else:
                lo = m
        pred.append(t)
        meas.append(0.5 * (lo + hi))

pred, meas = np.array(pred), np.array(meas)
ax[1].scatter(pred, meas, s=5, color="#1b6ca8", alpha=0.6, linewidths=0)
lim = [1.0, max(pred.max(), meas.max()) * 1.05]
ax[1].plot(lim, lim, "k--", lw=0.7)
ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel(r"predicted $\alpha^\star=\min_y b_y/(b_y-a_y)$")
ax[1].set_ylabel(r"measured crossover")
err = np.median(np.abs(meas - pred) / pred)
ax[1].set_title(rf"(b) crossover, median err {err:.2%}", fontsize=8)

plt.tight_layout()
plt.savefig("fig_ladder.pdf", bbox_inches="tight")
print(f"fig 1 done; n={len(pred)}, median rel err = {err:.4%}")

# ---------------------------------------------------------------- Figure 2 --
fig, ax = plt.subplots(figsize=(3.3, 2.3))
sig = []
for p_, m_ in zip(pred, meas):
    pass
rows = []
for ps in range(1, 60):
    s = np.random.default_rng(ps).permutation(k)
    if np.all(s == np.arange(k)):
        continue
    for p in np.linspace(-0.7, 0.7, 9) * np.pi / k:
        aa, bb, _, _ = one_x(k, p, 0, s)
        t = astar(aa, bb)
        if np.isfinite(t):
            keep = np.abs(aa - bb) > 1e-8
            rows.append((t, (bb[keep] - aa[keep]).max()))
R = np.array(rows)
ax.scatter(R[:, 0], R[:, 1], s=5, color="#d1495b", alpha=0.5, linewidths=0)
r = np.corrcoef(R[:, 0], R[:, 1])[0, 1]
ax.set_xlabel(r"$\alpha^\star$")
ax.set_ylabel(r"representational signal $\max_y (b_y-a_y)$")
ax.set_title(rf"trade-off, $r={r:+.2f}$", fontsize=8)
plt.tight_layout()
plt.savefig("fig_tradeoff.pdf", bbox_inches="tight")
print(f"fig 2 done; corr = {r:+.4f}, n={len(R)}")
