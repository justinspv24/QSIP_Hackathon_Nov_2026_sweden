#!/bin/bash
# Stage 3 GPU queue (one GPU, sequential). Waits for the K=8192 CD34+ measurement run first.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8
while pgrep -f "e13_stage3.py job --K 8192 --task CD34" > /dev/null; do sleep 10; done
for t in "CD19+ B" "CD4+/CD45RA+/CD25- Naive T" "CD4+ T Helper2"; do
  n=$(echo "$t" | tr '/ +' '__p')
  taskset -c 64-71 .venv/bin/python scripts/e13_stage3.py job --K 8192 --task "$t" --hl > "logs/stage3_K8192_${n}.out" 2>&1
done
taskset -c 64-71 .venv/bin/python scripts/e13_stage3.py job --K all --task "CD34+" > logs/stage3_Kall_CD34p.out 2>&1
