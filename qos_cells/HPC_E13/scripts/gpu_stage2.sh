#!/bin/bash
# Stage 2 pair-feature jobs on GPU 0, 2 at a time (gene-feature jobs run on CPU via e13_driver.py --filter genes)
cd "$(dirname "$0")/.."
mkdir -p logs/parts_stage2
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 NUMBA_NUM_THREADS=6
tr '\n' '\0' < logs/stage2_gpu_jobs.txt | xargs -0 -P 2 -I{} taskset -c 72-87 bash -c '{}'
