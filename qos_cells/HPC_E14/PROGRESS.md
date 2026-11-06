# E14 progress

Start 22:42 UTC (2026-11-05). Hard stop 04:42. Node: 8x RTX A6000, 128 cores, idle at start (nothing else running).

## Plan (measured)
| job | unit | measured time / unit | units | resources | wall |
|---|---|---|---|---|---|
| gate (E13 reproduction) | pair | 20 s (sklearn full ridge, 5 folds) | 46 | 46 procs | 51 s, done |
| m1 hashing + SGD | pair x S_h x seed x loss (16 hyper x 25 nested subsets x 5 epochs inside) | 7-9 s at 7.7k cells (scales with cells, not S) | 3,588 | 104 CPU procs | 9.2 min, done |
| m2 count-sketch selection + exact ridge | pair (116 rankings x 25 subsets x 22 k x 5 alpha) | measuring | 46 | 8 GPUs | running |
| ref: full ridge, alpha by nested CV | pair x alpha | ~4 s / fit | 276 | 100 CPU procs | running |
| m0 control: hashed exact ridge, alpha by nested CV | pair | - | 46 | 8 GPUs | after m2 |
| m3 Frequent Directions ridge | - | - | - | GPUs | Stage 2 |

## Status log
- 22:52 gate PASS: full ridge 0.901460 (E13 0.90146); E14 exact ridge engine on jl S=64 seed 0 = 0.808858 vs E13 0.808861.
- 23:04 m1 done. Finding: accuracy-tuned SGD beats the paper's FIXED alpha=200 ridge on accuracy, but with much lower
  balanced accuracy (0.72 vs 0.78 at S=64). Diagnosis: not a bug (SGD reaches ridge-level bacc when tuned on bacc);
  accuracy tuning trades class balance for majority-class accuracy. Fix of the comparison: add full ridge with alpha
  tuned by the same nested CV (fair ceiling) + hashed ridge with the same tuning (m0); report acc-tuned and bacc-tuned.
- 23:17 ref (full ridge, alpha by nested CV on accuracy) = 0.913 acc / 0.827 bacc; alpha=200 = 0.901 / 0.863. The
  fair accuracy ceiling for accuracy-tuned methods is 0.913, not 0.901.
- 23:30 found: running class weights n_seen/(2 n_seen(c)) are not exactly balanced (pair 23: 763 vs 628 instead of
  757.5/757.5) -> intercept bias to the majority class (hit FD hard: majority-only predictions). Fix for ALL streaming
  methods: class counts known up front (label-only pre-count, 2 ints) -> exact paper objective. FD with ell = 2d now
  equals exact ridge to 3 decimals. m1 re-running with known counts (old run kept: parts/m1_runningw).
- 23:31 FD: rows centred in-stream by Welford (exact centred scatter); robust-FD shrinkage correction tuned on/off.
- 23:35 M4 (RLS) = exact ridge on the same genes: 12/12 cases identical predictions (results/e14_m4_rls_check.json).
- m2 at ~17 min per 22k-cell pair per GPU; ETA ~00:05. m0 sharing GPUs. m3 next.
- 00:40 m2 done (88.9 min; slowed by GPU sharing). m3 done (46 pairs, all before 00:40). m0 paused 23:52-00:40 to
  prioritise m2, now resumed.
- Stage 1+2 snapshot (accuracy, nested-CV tuned on accuracy; S = 64):
  | method | (a) paper | (b) true floats |
  |---|---|---|
  | M1 hashing + SGD logistic, 1 pass | 0.873 | 0.861 |
  | M2 sketch selection + ridge, 2 pass | 0.756 | 0.746 |
  | M3 hashing + FD ridge, 1 pass | 0.811 | 0.716 |
  | E13 best data-oblivious (jl) | 0.809 | ~0.70 (S=8 hashed) |
  | full ridge alpha=200 / alpha nested | 0.901 / 0.913 | |
  M2 fails at small budgets because a count-sketch with C << D = 16,122 cannot rank genes (oracle exact top-k ridge:
  0.872 at k = 44, 0.876 at k = 64, but needs D counters to find them).
