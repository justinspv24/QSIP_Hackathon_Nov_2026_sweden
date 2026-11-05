#!/usr/bin/env bash
# Route 2 (E7) final assembly, run from the repo root:
#   bash scripts/09d_finalize_route2.sh results/e7_route2_K512.json results/e7_route2_K2048.json
# 1) merge the per-K result files into results/e7_route2.json  2) figure  3) backup slide + deck  4) markdown note
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/09_route2_classical_memory.py merge "$@"
python3 scripts/09_route2_classical_memory.py figure
python3 scripts/09b_patch_template_e7.py
python3 scripts/08_slides.py
python3 scripts/09c_route2_note.py > results/e7_route2_note.md
echo "[finalize] results/e7_route2.json · figures/fig_e7_memory.png · presentation/plan3.html · results/e7_route2_note.md"
