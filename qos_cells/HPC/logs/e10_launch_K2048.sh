#!/bin/bash
# E10 pairs at K=2048 (optional scope): one task per GPU; hashing+logreg off at this K (cut, see PROGRESS D15)
cd "$(dirname "$0")/.."
GPUS=(0 1 2 3)
for t in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${GPUS[$t]} .venv/bin/python -u scripts/e10_atlas.py run --task $t --space pairs --K 2048 --hl-S > logs/e10_task${t}_pairs_K2048.log 2>&1 &
done
wait
echo ALL_K2048_DONE
