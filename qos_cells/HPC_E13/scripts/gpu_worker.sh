#!/bin/bash
# One worker per GPU: takes the next job from logs/gpu_jobs_all.txt ("name|command"), atomic lock = mkdir logs/gpu_locks/<name>.
# A job is skipped if its lock exists (running/done elsewhere) or its stage-2 output exists.
# usage: gpu_worker.sh <gpu> <first core> <threads> [pids to wait for]
cd "$(dirname "$0")/.."
G=$1; C0=$2; T=$3; shift 3
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 10; done; done
export CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=$T NUMBA_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
while IFS='|' read -r name cmd; do
  [ -z "$name" ] && continue
  [ -f "logs/parts_stage2/${name}.json" ] && continue
  mkdir "logs/gpu_locks/${name}" 2>/dev/null || continue
  echo "$(date +%H:%M:%S) gpu$G start $name"
  taskset -c $C0-$((C0+T-1)) bash -c "$cmd"
  echo "$(date +%H:%M:%S) gpu$G end $name exit $?"
done < logs/gpu_jobs_all.txt
echo "$(date +%H:%M:%S) gpu$G queue empty"
