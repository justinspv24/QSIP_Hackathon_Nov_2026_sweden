#!/bin/bash
# waits for each prepped task, then runs gene + pair space at K=512 (one GPU per task)
cd "$(dirname "$0")/.."
GPUS=(1 2 3 5)
for t in 0 1 2 3; do
  until [ -f data/e10/task$t.json ]; do sleep 5; done
  sleep 3
  CUDA_VISIBLE_DEVICES=${GPUS[$t]} .venv/bin/python -u scripts/e10_atlas.py run --task $t --space genes --K 512 > logs/e10_task${t}_genes_K512.log 2>&1 &
  CUDA_VISIBLE_DEVICES=${GPUS[$t]} .venv/bin/python -u scripts/e10_atlas.py run --task $t --space pairs --K 512 > logs/e10_task${t}_pairs_K512.log 2>&1 &
done
wait
echo ALL_K512_DONE
