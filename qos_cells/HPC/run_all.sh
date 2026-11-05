#!/bin/bash
# Rerun E8–E11 on a multi-GPU node (tested on quant1: 8x RTX A6000, 128 cores, 2 TB RAM).
# Wall time on quant1: ~45 min end to end, dominated by the 14 GB AIDA download (~12 min) and its prep (~10 min).
# Usage:  bash HPC/run_all.sh            (from qos_cells/)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
if [ ! -x $PY ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
mkdir -p logs results figures data
run() { local gpu=$1; shift; local log=$1; shift; CUDA_VISIBLE_DEVICES=$gpu $PY -u "$@" > logs/$log.log 2>&1; }

# ---- E10 download (background, ~12 min) ------------------------------------------------------------------------
H5=data/aida_c838aec3.h5ad
URL=https://cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com/cell-census/2025-11-08/h5ads/c838aec3-03ef-4398-b882-0e3912abfff0.h5ad
( [ -f $H5 ] || curl -s -o $H5 $URL; sha256sum $H5 > data/aida_c838aec3.sha256 ) &
DL=$!

# ---- E8 (GPU 0), E11 (GPU 4), E9 (GPUs 1,2,5,6,7) -----------------------------------------------------------------
run 0 e8 scripts/e8_gpu_circuit.py run &
run 4 e11 scripts/e11_advantage_map.py run &
run 1 e9_K512_main   scripts/e9_fullpairs.py run --K 512 --seeds 5 --hl-seeds 5 --part main &
run 1 e9_K2048_main  scripts/e9_fullpairs.py run --K 2048 --seeds 5 --part main &
run 2 e9_K8192_main  scripts/e9_fullpairs.py run --K 8192 --seeds 5 --part main &
run 5 e9_K13714_main scripts/e9_fullpairs.py run --K 13714 --seeds 3 --part main --no-reservoir &
run 6 e9_K512_logreg  scripts/e9_fullpairs.py run --K 512 --part logreg --logreg-iter 300 &
run 6 e9_K2048_logreg scripts/e9_fullpairs.py run --K 2048 --part logreg --logreg-iter 200 &
run 7 e9_K8192_logreg scripts/e9_fullpairs.py run --K 8192 --part logreg --logreg-iter 200 &
run 3 e9_K13714_logreg scripts/e9_fullpairs.py run --K 13714 --part logreg --logreg-iter 200 &

# ---- E10 (after download): prep, then 4 tasks x {genes, pairs} at K=512 and pairs at K=2048 ----------------------------
wait $DL
$PY -u scripts/e10_atlas.py prep > logs/e10_prep.log 2>&1
G=(1 2 3 5); G2=(0 6 7 0)
for t in 0 1 2 3; do
  run ${G[$t]} e10_task${t}_genes_K512 scripts/e10_atlas.py run --task $t --space genes --K 512 &
  run ${G[$t]} e10_task${t}_pairs_K512 scripts/e10_atlas.py run --task $t --space pairs --K 512 &
  run ${G2[$t]} e10_task${t}_pairs_K2048 scripts/e10_atlas.py run --task $t --space pairs --K 2048 &
done
wait

# ---- merge, collect, figures (JSON only) --------------------------------------------------------------------------
$PY scripts/e9_fullpairs.py merge
$PY scripts/e10_atlas.py merge
$PY scripts/e11_advantage_map.py collect
$PY scripts/plots.py
echo "all done"
