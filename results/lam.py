"""
Figures for the lambda_max scaling law.

KEY FINDING: lambda_max is linear in M (R^2 = 0.993 over M = 2..32), but the
slope is NOT universal -- 0.488 on constructed models against 0.681 on
CIFAR-10. The slope is the mean pairwise correlation of the log-ratio
coordinates: for an equicorrelated matrix, lambda_max = 1 + (M-1) rho_bar, so
lambda_max / M -> rho_bar. Measured rho_bar tracks the slope closely
(0.484 vs 0.498 at M = 32).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.linewidth": .6,
                     "legend.frameon": False,
                     "xtick.major.width": .6, "ytick.major.width": .6})

# constructed: M, lam_max, sd, mean pairwise corr, n
R = np.array([
    [2, 1.875, 0.117, 0.874, 34], [3, 1.683, 0.208, 0.011, 40],
    [4, 2.130, 0.264, 0.237, 40], [5, 2.315, 0.404, 0.178, 35],
    [6, 2.876, 0.362, 0.271, 30], [8, 3.455, 0.608, 0.285, 32],
    [12, 5.391, 1.189, 0.380, 33], [16, 7.671, 1.244, 0.442, 29],
    [24, 11.670, 2.142, 0.463, 34], [32, 15.944, 3.020, 0.484, 34]])
CM = np.array([2, 3, 5]); CL = np.array([1.442, 1.904, 3.458])
CD = np.array([0.128, 0.065, 0.103])           # median d_fg, CIFAR
CC = np.array([57.8, 64.4, 13.3])              # certification rate %

sl_c = np.sum(R[:, 0] * R[:, 1]) / np.sum(R[:, 0] ** 2)
sl_f = np.sum(CM * CL) / np.sum(CM ** 2)

fig, ax = plt.subplots(1, 3, figsize=(6.9, 2.15))

# ---- (a) the scaling law ------------------------------------------------
xs = np.linspace(0, 34, 100)
ax[0].errorbar(R[:, 0], R[:, 1], yerr=R[:, 2], fmt='o', ms=3, lw=.8,
               capsize=2, color="#1b6ca8", label="constructed")
ax[0].plot(xs, sl_c * xs, color="#1b6ca8", lw=1, ls='--',
           label=rf"$\lambda_{{\max}}={sl_c:.2f}M$")
ax[0].plot(CM, CL, 's', ms=4, color="#d1495b", label="CIFAR-10")
ax[0].plot(xs, sl_f * xs, color="#d1495b", lw=1, ls=':',
           label=rf"$\lambda_{{\max}}={sl_f:.2f}M$")
ax[0].plot(xs, xs, color="0.7", lw=.7, label=r"$\lambda_{\max}=M$ (max)")
ax[0].set_xlabel("representation dimension $M$")
ax[0].set_ylabel(r"$\lambda_{\max}$")
ax[0].set_title("(a) linear in $M$, slope is not universal", fontsize=8)
ax[0].legend(fontsize=5.5, loc="upper left")
ax[0].set_xlim(0, 34)

# ---- (b) slope = mean correlation ---------------------------------------
ax[1].plot(R[:, 0], R[:, 1] / R[:, 0], 'o-', ms=3, lw=1, color="#1b6ca8",
           label=r"$\lambda_{\max}/M$")
ax[1].plot(R[:, 0], R[:, 3], '^--', ms=3, lw=1, color="#edae49",
           label=r"mean pairwise corr. $\bar\rho$")
ax[1].axhline(sl_f, color="#d1495b", lw=.8, ls=':',
              label="CIFAR-10 slope")
ax[1].set_xlabel("representation dimension $M$")
ax[1].set_ylabel("ratio")
ax[1].set_title(r"(b) $\lambda_{\max}\!\approx\!1+(M\!-\!1)\bar\rho$",
                fontsize=8)
ax[1].legend(fontsize=5.5, loc="lower right")
ax[1].set_ylim(0, 1)

# ---- (c) the ceiling, and where it binds --------------------------------
ax[2].plot(R[:, 0], 1 / (2 * R[:, 1]), 'o-', ms=3, lw=1, color="#1b6ca8",
           label="ceiling, constructed")
ax[2].plot(CM, 1 / (2 * CL), 's-', ms=4, lw=1, color="#d1495b",
           label="ceiling, CIFAR-10")
ax[2].plot(CM, CD, 'v--', ms=4, lw=1, color="#2e4057",
           label=r"observed median $d_{\mathbf{f},\mathbf{g}}$")
ax[2].fill_between(CM, CD, 1 / (2 * CL), where=(CD < 1 / (2 * CL)),
                   color="#2e4057", alpha=.10)
ax[2].set_yscale("log")
ax[2].set_xlabel("representation dimension $M$")
ax[2].set_ylabel(r"$d_{\mathbf{f},\mathbf{g}}$")
ax[2].set_title("(c) certifiable region closes", fontsize=8)
ax[2].legend(fontsize=5.5, loc="lower left")

plt.tight_layout()
plt.savefig("fig_lamscale.pdf", bbox_inches="tight")
print(f"fig_lamscale.pdf written; slopes {sl_c:.4f} (constructed), "
      f"{sl_f:.4f} (CIFAR)")

# ---- composition of pair outcomes ---------------------------------------
# A pie chart shows one composition. Three dimensions means three pies, which
# are hard to compare by eye -- a stacked bar puts them on a common axis.
fig2, ax2 = plt.subplots(1, 2, figsize=(6.6, 2.2))

cert = CC
below = np.array([100 - 57.8, 100 - 64.4, 100 - 13.3])   # uncertified
labels = [f"$M={m}$" for m in CM]
ax2[0].bar(labels, cert, color="#1b6ca8", label="certified (bound $<1$)")
ax2[0].bar(labels, below, bottom=cert, color="#d1495b", alpha=.75,
           label="not certified")
for i, v in enumerate(cert):
    ax2[0].text(i, v / 2, f"{v:.0f}%", ha="center", va="center",
                color="white", fontsize=7)
ax2[0].set_ylabel("share of seed pairs (\\%)")
ax2[0].set_title("(a) certification outcome by dimension", fontsize=8)
ax2[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(.5,-.18), ncol=2)
ax2[0].set_ylim(0, 100)

w = [57.8, 64.4 - 57.8, 0]
ax2[1].pie([57.8, 42.2], labels=["certified\n57.8%", "not certified\n42.2%"],
           colors=["#1b6ca8", "#d1495b"], startangle=90,
           wedgeprops=dict(width=.45, edgecolor="w"),
           textprops=dict(fontsize=7))
ax2[1].set_title("(b) $M=2$: refined vs naive constant\n(naive certifies 0%)",
                 fontsize=8)
plt.tight_layout()
plt.savefig("fig_certification.pdf", bbox_inches="tight")
print("fig_certification.pdf written")