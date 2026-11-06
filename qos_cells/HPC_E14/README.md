# E14: Can stronger classical streaming learners close the paper's PBMC68k gap?

**Summary**
- Best at S ≈ 44–64 (paper accounting): **0.861 / 0.873**, from hashing + one-pass logistic SGD (M1). True floats: 0.851 / 0.861. E13 best data-oblivious sketch: ≈ 0.81 at 64.
- S needed to get within 0.01 of full ridge (0.901), paper accounting / true floats: M1 **256 / 512**; M1b (AdaGrad) 256 / 512; M2 8,192 / not reached; M3 not reached / not reached; E13 sketches ≈ 8,192 / ≥ 3×10⁷.
- Fair ceiling: full ridge with α tuned by the same nested CV = **0.913**. At S = 64, M1 is 0.04 below it.
- Tuned on balanced accuracy, M1 at S = 64 = 0.766 bacc, close to E13's hashed ridge (≈ 0.78). Part of the accuracy gain comes from accuracy tuning trading class balance.
- Control M0 (paper sketch + exact ridge, tuned the same way) at S = 64: 0.811, the same as E13 (0.809). Re-tuning α does not explain SGD's gain; SGD's extra accuracy comes with *lower* balanced accuracy than exact ridge (0.766 vs 0.784).
- **Verdict: gap narrowed, not closed.** No method reaches full-ridge accuracy at S ≈ 44–64; the best is ~0.04 below the fair ceiling.

Figure: `figures/fig_e14_gap.png` (accuracy). `figures/fig_e14_gap_bacc.png` (tuned on and scored by balanced accuracy).

## What was tested
Protocol as E13 Stage 4, reused unchanged:
- **Data:** scvelo PBMC68k (sha256 c93b1c…, the E13 download), log1p, D = 16,122 genes.
- **Pairs and folds:** the paper's 46 unique pairs, weighted by draw count; StratifiedKFold(5) with no shuffle; metric = CV accuracy, with balanced accuracy stored too.
- **Gate:** full ridge reproduced exactly (0.901460). The E14 exact ridge engine matches E13 sklearn on signed hashing at S = 64 (0.808858 vs 0.808861).

Tuning:
- **Nested CV on the training folds only.** For outer fold k, the inner folds are the 4 other outer blocks.
- **Tuned by nested CV:** all hyperparameters (η, λ, α, loss, plain vs averaged, hash size, counter/k split, (d, ℓ), robust-FD on/off), chosen under each memory budget.
- **Main runs** are tuned on accuracy; `__tuned_on_bacc` variants are stored too.
- **Seeds:** 3 hash/sketch seeds (the paper's seeds for hashing).

### Memory accounting
- **The paper's own count** is in `pbmc68k_svm.py`: `space_streaming = feature_dim`. Ridge on S hashed features counts as **S**, and the S×S covariance is **not** counted.
- **The paper's adaptive survey** (`survey_adaptive.py`) counts all adaptive state (counters, ids, values) but not per-step workspace.
- **(a) paper accounting:** model registers plus adaptive state (counters, ids, averaged copy, AdaGrad accumulators). Gram/covariance and per-step workspace are not counted.
- **(b) true peak floats:** everything kept at any time: weights, bias, Gram k(k+1)/2, weighted sums, counters, ids, and scalars.
- **Plotting:** each point sits at the mean memory of the configs actually chosen.

## Methods

| id | method | passes | (a) | (b) |
|---|---|---|---|---|
| M1 | signed hashing (paper 'jl') + SGD, squared or logistic loss, normalised step η0/((q̄+1)√t), plain or Polyak-averaged (lazy L2 and lazy averaging) | 1 (5 epochs flagged) | S (2S averaged) | +6 (+8) |
| M1b | as M1 with AdaGrad (VW default); setup check for M1 | 1 (5 flagged) | 2S | 2S+4 |
| M2 | count-sketch (1 or 3 rows) of class sums → score \|Δ\| or t → top-k heap → exact balanced ridge on k genes | 2 | C+2k | max(C+2k+3, k(k+1)/2+3k+4) |
| M4 | recursive least squares on the M2 genes | 2 | = M2 | ≈ M2 |
| M3 | hashing to d + Frequent Directions (rank ℓ; Welford-centred rows; robust-FD optional) + ridge | 1 | d | ℓd+2d+4 |
| M0 | control: hashing + exact ridge, same nested tuning as above | 1 | S | S(S+1)/2+2S+3 |
| ref | full ridge, all genes; α = 200 (paper) or α by the same nested CV | 1 | – | – |

- **Class weights:** every streaming learner uses the paper's balanced weights n/(2n_c), with the class counts known up front (a label-only pre-count, 2 integers).
- **M4 = M2:** RLS gave identical predictions to exact ridge in 12/12 checks (`results/e14_m4_rls_check.json`). So M4 shares M2's curve.
- **Method 5 not run.** Instead, AdaGrad (M1b) was added as a setup check on M1 (one-line reason: VW's default optimiser is the strongest well-known form of method 1).

## Results (accuracy, tuned on accuracy; 46 pairs weighted, 3 seeds)

| method | S=44 (a) | S=64 (a) | S=64 (b) | S=256 (a) | S=1024 (a) | S=8192 (a) | bacc-tuned bacc @64 (a) | S* (a) / (b) |
|---|---|---|---|---|---|---|---|---|
| M1 logistic, 1 pass | **0.861** | **0.873** | **0.861** | 0.902 | 0.912 | 0.917 | 0.766 | 256 / 512 |
| M1 squared, 1 pass | 0.858 | 0.870 | 0.858 | 0.900 | 0.911 | 0.917 | 0.772 | 256 / 512 |
| M1 logistic, 5 epochs [multi-pass] | 0.855 | 0.868 | 0.855 | 0.902 | 0.913 | 0.918 | 0.775 | 256 / 512 |
| M1b AdaGrad logistic, 1 pass | 0.843 | 0.859 | 0.843 | 0.900 | 0.917 | 0.924 | 0.740 | 256 / 512 |
| M2/M4 sketch selection + ridge [2 passes] | 0.743 | 0.756 | 0.746 | 0.836 | 0.877 | 0.896 | 0.528 | 8192 / – |
| M3 hashing + FD ridge | 0.786 | 0.811 | 0.716 | 0.858 | 0.867 | 0.869 | 0.784 | – / – |
| M0 control (hashing + exact ridge, nested α) | 0.791 | 0.811 | – (needs ≥171) | 0.863 | 0.890 | 0.910 | 0.784 | 2048 / – |
| E13 jl + ridge, α = 200 | – | 0.809 | ≈0.70 (S=8) | 0.852 | 0.872 | 0.894 | 0.779 (α=200) | 8192 / ≥3e7 |
| full ridge, all genes: α = 200 / α nested | | | | | | | 0.863 / 0.869 | 0.901 / 0.913 |

- **S\*** is the smallest S in the grid from which accuracy stays ≥ 0.891 (full ridge 0.901 − 0.01). It is computed on the budget grid S ∈ {16, 32, 44, 64, 128, …, 8192}.
- **Oracle check for M2:** exact top-k genes (all D class statistics, *not* memory-limited) + ridge gives 0.872 at k = 44 and 0.876 at k = 64. So M2's failure is in *finding* the genes: a count-sketch with C ≪ D cannot rank them.

## Interpretation (what the data supports)
1. **Learner choice matters a lot at S ≈ 64.** One-pass hashed SGD (0.87) beats every E13 data-oblivious sketch with exact ridge (≈ 0.81). Its memory is honest: (a) ≈ (b).
2. **The gain is an accuracy/balance trade-off, not a better use of memory.**
   - With α tuned the same way, exact ridge on the same hashed features stays at 0.811 (M0), so tuning α alone does not close the difference.
   - SGD's accuracy-tuned solutions shift predictions toward the majority class: higher accuracy, lower balanced accuracy (0.766 vs 0.784 for M0 at S = 64).
   - On balanced accuracy, no new method beats the E13-type hashed ridge at S ≈ 64.
   - The paper's α = 200 ridge is also not accuracy-optimal: the full ridge with α tuned the same way reaches 0.913.
3. **Covariance-based learners (exact ridge, M2, M3) are expensive under true floats.** Their Gram is O(k²) or O(ℓd). Under (b), only SGD-type learners are competitive at small budgets.
4. **Supervised selection needs counters ≈ D.** With C ≪ D, count-sketch ranking is close to random.

## Not tested / limits
- Only PBMC68k and the paper's 46 pairs; only linear models.
- Hash seeds and stream orders: 3 each.
- The quantum machine size (42–44) is the paper's formula only. Nothing quantum was run, and no claim is made about quantum advantage or about the paper being wrong.
- Running (stream-estimated) class weights were tried first and gave an intercept bias. They were replaced by known class counts (`parts/m1_runningw` keeps the old M1 run).
- (b) for E13 truncation excludes the D gene-frequency counters that truncation itself needs.

## Files
- `results/e14_gate.json`: E13 reproduction.
- `results/e14_stage1.json`: all curves (a/b, tuned on acc/bacc), S*, references, config, git hash, hardware.
- `results/e14_m4_rls_check.json`: RLS = exact ridge.
- `scripts/`: e14_core (shared), e14_m1_sgd, e14_m1b_adagrad, e14_m2_select, e14_m3_fd, e14_m0_hashridge, e14_ref_full, e14_merge, e14_figures, e14_gate.
- `PROGRESS.md`: timeline, wall times, the setup bugs found and fixed.
