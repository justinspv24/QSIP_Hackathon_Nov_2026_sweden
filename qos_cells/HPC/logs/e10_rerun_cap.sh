#!/bin/bash
# D17: rerun the runs that hit the 4e9 records/seed cap, with cap 1e11
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/e10_atlas.py run --task 2 --space pairs --K 512 --max-records 1e11 --time-cap 60 > logs/e10_task2_pairs_K512_cap1e11.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/e10_atlas.py run --task 2 --space pairs --K 2048 --max-records 1e11 --time-cap 60 --hl-S > logs/e10_task2_pairs_K2048_cap1e11.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/e10_atlas.py run --task 1 --space pairs --K 2048 --max-records 1e11 --time-cap 60 --hl-S > logs/e10_task1_pairs_K2048_cap1e11.log 2>&1 &
wait
echo RERUN_DONE
