# FINDINGS

Every measured result, with the script that produces it. Numbers marked
**(real)** come from trained networks; the rest from constructed models.

---

## 1. Rate lemma and the critical order (Thm 5)

`theory/thm05_rate_lemma.py`, `theory/thm05_rate_prediction.py`

| check | predicted | measured | rel. err |
|---|---|---|---|
| crossover, forward | 1.349915 | 1.349919 | 2.8e-6 |
| crossover, reverse | 1.276277 | 1.276430 | 1.2e-4 |
| crossover, 295 geometries | — | — | **median 0.00%** |
| exponential decay rates | — | — | < 2.3% worst |
| linear growth slopes | — | — | < 0.06% worst |
| rate prediction, 739 (geometry, order) pairs | — | — | 2.3% median, **2.32 decades** |

Correction required: labels with `a_y == b_y` cancel exactly
(`1 - alpha + (alpha-1) = 0`) and must be excluded. Omitting this inflates
predicted rates by 40–70%. At `alpha = 1` all first-order terms cancel, giving
KL a polynomial prefactor: `d_KL = Theta(rho * exp(-c rho a_min))`.

## 2. alpha*_sym ceiling

`theory/thm15_topological.py`, `theory/harness.py`, `experiments/constructed/`

- **Exhaustive, k=6, all 720 permutations:** rotations give `alpha*_sym = inf`
  and `d_fg = -0.000`; reflections 1.109; order-breaking 1.047.
  **both-large = 0 in every class.**
- Non-uniform norms do not help: order-breaking caps at 1.06 rather than 1.09.
- Violation count is irrelevant: median `alpha*_sym` is flat at 1.044 across
  2–8 cyclic-order violations. A discrete invariant, not a continuous penalty.
- **M-scaling** (200 geometries per M, margin floor 0.08, `d_fg >= 0.10`):
  best `alpha*_sym` = 1.34, 1.31, 1.20, 1.26, 1.18, 1.22 at M = 2,3,4,6,8,12;
  medians ≈ 1.00. **Ceiling ≈ 1.35, saturating in M.**

## 3. Dichotomy (Thm 15)

`theory/thm15_dichotomy.py`

`alpha*_sym = inf  <=>  p = p'  <=>  ~_L equivalence`.

Regression over 329 pairs: `max |d_fg| = 5.0e-4` in the infinite class against
`min |d_fg| = 6.5e-2` in the finite class. The residual decays as **O(1/n)**
(2.0e-3 at n=500 → 6.3e-5 at n=16000), the finite-sample bias of the
cross-covariance estimator. CI tolerance is therefore `5/n`, not a constant.

## 4. Perturbation bound constants (Thm 12, Prop 16)

`theory/thm12_constants.py`, `theory/thm12_tightening.py`

Where the slack lives, measured:

| step | gain |
|---|---|
| RMS over coordinates instead of worst-case | 1.35–1.57x |
| keep `tau_l = |s_l - s'_l|` separate from `sd_l` | median `tau/sd` = 0.25–0.53 |
| **combined** | **1.65–2.37x (1.79x at M=2,3)** |
| spectral form `sqrt(2 lam (1-rho_bar))` vs Prop-13 floor | within 11–17% |

The spectral step is essentially tight; all residual looseness is in passing
from log-ratio perturbation to coordinate correlations.

Validity: 0/139 violations on constructed models, 0/45 **(real)**.

## 5. Structural ceilings (Prop 13, Rem 17)

`theory/prop13_ceilings.py`, `theory/thm14_delta_floor.py`

- `sum_l Var[z'_l - w'_l] >= 2M d_SVD` (trace ≤ nuclear norm), so
  `bound >= sqrt(2 lam_max d_SVD)` and **`bound < 1` forces
  `d_fg < 1/(2 lam_max)`**.
- Cor 19 requires `alpha > 4 sqrt(M) Delta/psi_min`; measured
  `Delta/psi_min = 11.34` **(real)** gives `alpha > 64` against a threshold
  near 1.35 — **48x out of reach**.
- KL's floor is not repairable: `p_(M+1) <= 1/(M+1)` caps Thm 14 at
  `eps < 1.7e-3` for *any* model. Quantile floors buy 9 orders and need 13
  more **(real)**: discarding the worst 10% of inputs lifts `1/delta_3` from
  3.56e14 to 4.79e5, still demanding `eps < 6.8e-14`.

---

## 6. Synthetic angular classification **(real)**

`experiments/synthetic/train_synth.py`, `experiments/analyze.py`
— 400 models, 3800 pairs, 0 dropped, 15k steps.

**Width dissociation.** Training loss flat in width (0.0202→0.0208 at c=4), so
width and fit quality are separated. Then:

| c | mean KL (16→256) | mean d_fg (16→256) | loss-adj. slope |
|---|---|---|---|
| 4 | 0.116 → 0.065 | 0.251 → 0.366 | +0.034 |
| 6 | 0.788 → 0.349 | 0.475 → 0.558 | +0.028 |
| 10 | 5.06 → 1.65 | 0.436 → 0.697 | +0.098 |
| 18 | 16.1 → 16.6 | 0.425 → 0.733 | +0.073 |

**Wider networks learn closer distributions and more dissimilar
representations.** Confirmed under mCCA (the measure prior work uses), after
the >=90% accuracy filter (315/400 retained):

| c | mCCA 16 → 256 | pairs retained |
|---|---|---|
| 4 | 0.741 → 0.666 | 190 at every width (clean) |
| 6 | 0.639 → 0.530 | 190, except 171 (clean) |
| 10 | 0.733 → 0.437 | 136–190 (uneven) |
| 18 | 0.833 → 0.451 | 45, 3, 0, 10, 28 (**unusable**) |

Claim rests on c=4 and c=6, where the filter removes essentially nothing.

Other: C3 non-vacuity 1.3% at median `d_fg = 0.570` (the ceiling binds here,
unlike CIFAR); C4 agreement +0.541 / -0.331 / -0.302 / +0.276;
`delta_3 = 4.2e-74`; argmax agreement min 0.477, median 0.969.

## 7. CIFAR-10 **(real)**

`experiments/cifar/train.py`, `experiments/analyze.py`, `experiments/cifar/dimcheck.py`
— 60 models (M ∈ {2,3,5} × 10 seeds, plus M=2 smoothing arm), 270 pairs, 0 dropped.

Mean test accuracy 0.737 / 0.840 / 0.867 at M = 2 / 3 / 5.

**Temperature intervention (the causal test).** Rescaling logits by `1/T`
leaves `d_fg` invariant to machine precision while `delta_3` climbs:

| T | delta_3 | mean KL | mean d_fg | Spearman |
|---|---|---|---|---|
| 1 | 7.9e-11 | 0.601 | 0.1918 | 0.725 |
| 2 | 8.5e-6 | 0.409 | 0.1918 | 0.766 |
| 4 | 2.5e-3 | 0.289 | 0.1918 | 0.787 |
| 8 | 3.5e-2 | 0.183 | 0.1918 | 0.792 |
| 16 | 9.6e-2 | 0.096 | 0.1918 | **0.816** |

`d_fg` invariance check: **max change 0.00e+00**.

**Label smoothing is confounded** — a negative control. Accuracy falls
0.737 → 0.633 and `d_fg` moves (0.192, 0.153, 0.053, 0.143); correlations are
non-monotone (0.725, 0.640, 0.338, 0.911). No conclusion drawn.

**Certification, per dimension:**

| | M=2 | M=3 | M=5 |
|---|---|---|---|
| lam_max | 1.442 | 1.904 | 3.458 |
| ceiling 1/(2 lam_max) | 0.347 | 0.263 | 0.145 |
| median d_fg | 0.128 | 0.065 | 0.103 |
| median bound, naive | 1.558 | 1.794 | 2.904 |
| median bound, refined | **0.931** | **0.764** | 1.226 |
| non-vacuous, naive → refined | 0% → **58%** | 0% → **64%** | 0% → 13% |
| violations | 0 | 0 | 0 |

`lam_max ≈ 0.68 M`, so certification should fail outright once typical `d_fg`
exceeds ≈ `0.74/M` — near M ≈ 7 for networks of this kind.

**Both failure modes track dimension together.** KL's correlation with `d_fg`
falls 0.725 → 0.722 → **0.143 (p=0.35, n.s.)** over M = 2,3,5, while
certification falls 58% → 64% → 13%. Different mechanisms (`delta_(M+1)`
reaching deeper into the tail; `lam_max` growing), same direction.

Other: `delta_(M+1)` = 1.1e-15 / 8.8e-15 / <1e-16; top-k recovers 44/65/89% of
full KL at k=2,3,5 for M=2, rising to 67/84/95% at M=5 (k=1 is identically
zero by construction); argmax agreement min 0.607, median 0.804.

---

## Known issues, carried in the paper

1. The synthetic arm does **not** apply prior work's >=90% accuracy filter at
   training time; 20/100 c=18 models fall below 0.80 and one c=6 model reaches
   0.825.
2. `analyze.py`'s C1 block aggregates `delta_m` across class counts rather than
   conditioning on them.
3. C4 is reported pooled over dimensions; the magnitudes fall to 0.36–0.60
   because of the dimension effect, not measure disagreement.
4. Argmax agreement reaches a minimum of 0.477 (synthetic) / 0.607 (CIFAR), so
   `alpha*_sym` on those pairs rests on a reduced subset.

## Results that were retracted

- "Breaking circular symmetry lowers `alpha*`" — survivorship bias; the
  shared-argmax gate rejected 120/120 configurations. See
  `experiments/constructed/diag.py`.
- All `alpha*` values computed in one direction only; the symmetric definition
  changed 1.62/1.67/1.88 to ≤ 1.35.
- "The `delta_m` floor is repairable by a quantile" — buys 9 orders, needs 13.
- "Prop 13's ceiling blocks certification on CIFAR-10" — it does not bind;
  a 1.67x constant did.
