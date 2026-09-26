#!/usr/bin/env bash
# Sections 7.2, 7.3, 7.4. One analysis pipeline for all three.
set -euo pipefail
OUT=${1:-runs}

echo "== step 0: smoke tests (about 3 minutes; do not skip) =="
python test_measures.py
python train_synth.py --smoke --out /tmp/smoke
python train_lm.py    --smoke --out /tmp/smoke
python train.py       --smoke --out /tmp/smoke   # needs CIFAR-10; see README

echo "== 7.2 synthetic: width sweep, 20 seeds, FULL 15k-step budget =="
mkdir -p "$OUT/synth"
for C in 4 6 10 18; do for W in 16 32 64 128 256; do for S in $(seq 0 19); do
  python train_synth.py --classes $C --width $W --seed $S --out "$OUT/synth"
done; done; done
echo "== 7.2 smoothing arm =="
for SM in 0.05 0.1 0.2; do for S in $(seq 0 19); do
  python train_synth.py --classes 6 --width 64 --seed $S --smooth $SM --out "$OUT/synth"
done; done

echo "== 7.3 CIFAR-10: 10 seeds x 3 dims, then the smoothing arm =="
mkdir -p "$OUT/cifar"
for D in 2 3 5; do for S in $(seq 0 9); do
  python train.py --dim $D --seed $S --smooth 0.0 --out "$OUT/cifar"
done; done
for D in 2 3 5; do for SM in 0.05 0.1 0.2; do for S in $(seq 0 9); do
  python train.py --dim $D --seed $S --smooth $SM --out "$OUT/cifar"
done; done; done

echo "== 7.4 language model: 4 seeds =="
mkdir -p "$OUT/lm"
for S in 0 1 2 3; do
  python train_lm.py --seed $S ${LM_TEXT:+--text "$LM_TEXT"} --out "$OUT/lm"
done

echo "== analysis (run each separately; pairs are only meaningful within a setting) =="
mkdir -p results
for K in synth cifar lm; do
  python analyze.py --runs "$OUT/$K" --out "results/$K" | tee "results/$K.txt"
done
