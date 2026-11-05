#!/bin/bash
# Stage 1, pairs K=2048, on the GPU path (replaces the slow CPU jobs; decision D11 in PROGRESS.md)
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
for t in "CD34+" "CD4+ T Helper2" "CD4+/CD45RA+/CD25- Naive T" "CD19+ B"; do
  n=$(echo "$t" | tr '/ +' '__p')
  taskset -c 0-7 .venv/bin/python scripts/e13_stage3.py job --K 2048 --task "$t" --hl --seeds 3 --out "logs/parts_stage1/pbmc68k_pairs2048_${n}_split0.json" > "logs/stage1_gpu_K2048_${n}.out" 2>&1
done
