# CIFAR-10 experiments (C1–C4)

## What is tested and what is not

`measures.py` is pure numpy and **is tested** — `python test_measures.py`
covers every quantity the paper reports, including the identity cases
(`d_SVD(Z,Z)=0`, `d_fg=0` on a `~_L`-equivalent pair) at the `5/n` tolerance the
CI rule requires.

`train.py` **could not be executed by the author** (no GPU, no dataset access in
the authoring environment). It is written to the protocol of the prior work but
should be smoke-tested first:

```bash
python train.py --dim 2 --seed 99 --smoke --out /tmp/smoke
```

50 steps on one batch, ~1 minute, catches shape/dtype errors before you commit
40 GPU-hours. The two places most likely to need adjustment on your setup are
the dataloader `num_workers` and the `--lr`, which the prior work does not
state precisely.

`analyze.py` **is tested end-to-end** on mock runs (`runs_mock/`), which is how
two real bugs were found: `0 × (-inf)` NaNs in the restricted KL, and each model
being restricted to its *own* top-k support, which makes the KL ill-defined
whenever the supports differ. Both fixed; `topk_pair`/`tail_pair` now restrict
both models to the **reference model's** support.

## Running

```bash
./run_all.sh runs        # smoke test, then train, then analyse
```

Or in pieces — `train.py` writes one `.npz` per model, `analyze.py` reads the
directory, so training happens once and analysis is cheap and re-runnable.

## Protocol

| exp | question | output |
|-----|----------|--------|
| C1 | Does `delta_m` stay catastrophic on real models? Is KL saturated by the top-1 label? | `delta_m` profile vs `m`; KL on top-k vs tail supports |
| C2 | Does raising `delta_m` by label smoothing restore KL's predictive power? | Spearman(`d_KL`, `d_fg`) vs smoothing |
| C3 | Is Thm 12's bound non-vacuous at M = 2, 3, 5? | violations and non-vacuity fraction per dim |
| C4 | Do the findings survive a change of similarity measure? | Spearman with KL for `d_fg`, mCCA, CKA, Procrustes |

**C1 is the one that matters most.** On synthetic data `delta_m` was
5.2e-93 without smoothing and the subset fix bought only ~17x. If CIFAR-10
classifiers place genuine mass on two or three confusable classes, Remark 16's
pessimism is too strong and Theorem 14 should be sold differently. Run C1 first.

## Guards

`analyze.py` reports, never silently drops:

- `dropped` pair count — **must be 0**, or be explained in the paper;
- argmax agreement per pair — low values mean `alpha*_sym` rests on a small
  subset and the number is not trustworthy;
- `alpha*_sym = inf` pairs must satisfy `d_fg < 5/n` (Theorem 15). Any
  violation is a real bug, not noise.

Two of the three substantive errors in this project were silent filters. Read
these lines before reading the results.

## Compute

Roughly 27 GPU-hours for the 30 unsmoothed models plus ~13 for the smoothing
arm, on one A5000. Analysis is CPU-only, a few minutes.

---

# Update: all three sections, one pipeline

`train_synth.py` (7.2), `train.py` (7.3) and `train_lm.py` (7.4) all emit the
same `.npz` schema, so `analyze.py` serves all three. Run the analysis
**separately per setting** — cross-setting pairs are meaningless.

```bash
./run_all.sh runs                 # smoke tests, then 7.2, 7.3, 7.4, then analysis
LM_TEXT=corpus.txt ./run_all.sh   # use a real corpus for 7.4
```

## Why 7.2 must be re-run here

The authoring-environment run used **3000 steps**, which left wide models
undertrained: they carried *higher* loss than narrow ones, the reverse of the
usual regime, and the width result was uninterpretable. `train_synth.py`
defaults to **15000 steps and 20 seeds**, matching the prior work. The
loss-adjusted regression in `analyze.py` reports naive and adjusted slopes side
by side; a **negative loss-adjusted slope** is the claim that wider networks
learn more similar representations. The authoring run gave `+0.062` naive,
`+0.025` adjusted — do not quote those, they are from the broken regime.

## If the CIFAR-10 download is blocked

`torchvision.datasets.CIFAR10` fetches from `www.cs.toronto.edu`. If that host
is unreachable:

1. Download `cifar-10-python.tar.gz` anywhere you have access and place it at
   `./data/cifar-10-python.tar.gz`. torchvision extracts it and skips the
   download; `--download=True` is then a no-op.
2. Or point `--data` at an existing copy on the cluster — many shared
   filesystems already have one under a datasets mount.
3. Verify before the long run:
   `python -c "from torchvision import datasets; datasets.CIFAR10('./data', download=True)"`

The MD5 torchvision checks is `c58f30108f718f92721af3b95e74349a`.

## 7.4 notes

The LM uses **tied embeddings**, so the unembedding matrix *is* the input
embedding and the model is literally Eq. (1). Diversity needs `V > M`; the
script prints `OK`/`FAILS` at startup. With no `--text` it falls back to a
synthetic Markov corpus so the script always runs — fine for a smoke test,
not for the paper.

This is the sharpest test of Cor. 15: LMs place real mass on only a few next
tokens, so `delta_m` should be tiny and KL should be least informative there.
Check the `C1` block of `results/lm.txt` first.

## Suggested order

1. `test_measures.py` and the three `--smoke` runs — 3 minutes, catches
   everything cheap.
2. **7.3 C1 only** (10 seeds at `--dim 2`, ~9 GPU-h): gives the real
   `delta_m` profile. If it is not catastrophic, Remark 16 is too pessimistic
   and part of the paper needs rewriting before the rest of the compute is
   spent.
3. 7.2 full (cheap, CPU-feasible).
4. The rest of 7.3, then 7.4.
