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

## 20:38 — status 1 (T+38 min)

| stage | state | key numbers | ETA |
|---|---|---|---|
| 1 | 6/12 jobs done; pairs K=2048 (CPU) running | exact centroid BA: B 0.85, CD34+ 0.82–0.84, naive CD4 0.73–0.74, Th2 0.55–0.59 (no signal). pairs K=512 CD34+: best S*/N 1.2e-4. genes K=1024: B S*/N 1.0, naive CD4 0.5 | 21:20 |
| 3 | K=8192 CD34+ done (GPU), other tasks queued | GPU Gram path = materialised pairs (rel. L2 3e-7, same BA); logreg all N BA 0.94 | ~22:30 |
| 4 | script ready, 1-pair test OK | paper's own file (scvelo, 65,877 × 33,939, sha256 c93b1cca…), D = 16,122, 46 unique pairs of the paper's 100 draws | after 2 |

Decisions:
- D6. E8–E11 finished at 20:16 (final status, GPUs idle), so the 100-core / GPU budget applies from 20:20. One GPU (device 0) is used.
- D7. Stage 3 started on the idle GPU while Stage 1 runs on CPU (does not slow Stage 1). Stage 1 is still committed first.
- D8. Genes K=1024 B: S* = N is real, not a setup bug: sketches degrade smoothly (hashing S=128: 0.830 vs exact 0.848), the band is ±0.01 (1,772 test positives). A ±0.02 sensitivity column (`S_star_tol002`) is added.
- D9. Stage 2: no full-N logreg (reference only, in Stage 1), record cap 1e9 per seed (M→∞ variant still gives a converged reference), hashing+logreg via exact folding of the S_max hashed features (checked: max diff 7e-8).
- D10. No push possible from this node (no GitHub credentials, same as the hpc session). All stages are committed locally on hpc-e13.

## 21:15 — Stage 1 done

- D11. Pairs K=2048 moved from CPU (would finish ~22:00) to the GPU path (E9 pairs_gpu.py, gated vs materialisation). CPU logs in logs/superseded/. M→∞ and hashing+logreg use 1 seed on this path.
- Result: best one-pass S*/N ≤ 3.9e-3 on all pair tasks (S* 8–512); genes K=1024 up to 1.0 (B). Th2 no signal. Matches PBMC3k.
- Peak RAM 36.8 GB (CPU job), sum of job wall time 2.0 h.
