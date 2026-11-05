# E13 progress (PBMC68k, branch hpc-e13)

Start 20:00 UTC, hard stop 06:00 UTC. Times are UTC.

## 20:20 — setup done, Stage 1 started

| item | state |
|---|---|
| data | PBMC68k fresh donor A (10x, Cell Ranger 1.1.0) + Zheng et al. per-cell annotation file; sha256 in `results/e13_data.json` |
| cells | 68,579, all used (no subsampling) |
| tasks (prevalence) | CD4+ T Helper2 0.14 %, CD34+ 0.40 %, CD4+/CD45RA+/CD25- Naive T 2.7 %, CD19+ B 8.6 % |
| checks | stream sampler = E7 law (z-rms 1.04, var ratio 1.005); fast pair builder = E7 exactly; exact centroid + logreg reproduce E7 on PBMC3k DC (0.9193 / 0.9545) |
| Stage 1 | 12 jobs (3 feature sets × 4 tasks, 3 seeds) on 16 cores, ETA ~21:30 |

Decisions:
- D1. All 68,579 cells used: nnz is only 37 M, no subsampling needed.
- D2. Stream sampled as R ~ Poisson(λ·nnz) records on uniform nonzero entries (same law as E7, O(records) instead of O(nnz)).
- D3. Extra S* variant "M→∞": each sketch applied to the exact Δ. E7's protocol (sketch of the stream at M_run) stays the main number; the M→∞ one separates memory from sampling noise when the stream cannot converge in the record cap (3·10⁹ per seed on CPU).
- D4. Th2 (97 cells) has no signal with these features (K=2048 pairs: logreg BA 0.50, centroid 0.55). Kept and flagged `signal: false` (exact BA < 0.6); its S* is not informative.
- D5. Reservoir and dense JL dropped (not in the E13 method list).
