# E13 backup slides (one per stage)

## Stage 1 — On PBMC68k, a few hundred classical counters match the exact centroid: no memory advantage
- Gene pairs, N = 1.3·10⁵ – 2.1·10⁶ (18–22 index qubits): best one-pass sketch S* = 16–512 counters (S*/N ≤ 3.9·10⁻³, ≤ 4 kB). PBMC3k gave 8–1,024.
- On 1,024 genes, S*/N is up to 1.0 for B (strict ±0.01 band, small N), still ≤ 8 kB. Th2 (0.14 %) has no signal (BA ≤ 0.60).
- The centroid is weak on this data (B: 0.85 vs logreg 0.90–0.95). The test covers the sketchable classifier only.
- Figure: `HPC_E13/figures/fig_e13_sstar_vs_N.png`

## Stage 2 — The "few hundred counters" result holds on all 5 splits, on PBMC68k and PBMC3k alike
- 5 splits × 5 seeds, 105 jobs. Gene pairs: best one-pass S*/N ≤ 0.016 on every split (K=2048: ≤ 4.9·10⁻⁴, ≤ 1,024 counters).
- PBMC68k needs 2–10× more relative memory than PBMC3k (B: ~900 vs ~180 counters): same order, far below N.
- On 1,024 genes S*/N swings 0.03–1.0 between splits (small N, tight band), still ≤ 8 kB.
- Figure: `HPC_E13/figures/fig_e13_robustness.png`

## Stage 3 — Even at N = 1.5·10⁸ gene pairs (28 qubits), ≤ 4,096 classical counters match the exact centroid
- K = 8,192 (N = 3.4·10⁷) and all ≈ 17,100 expressed genes (N = 1.46·10⁸), centroid via Gram matrix, never materialised (checked vs materialisation at K=512).
- Best one-pass S* = 512–4,096 counters (S*/N ≤ 3.1·10⁻⁵). Flat for B and naive CD4 T; grows ~∝ N for CD34+ (S*/N ≈ 2–3·10⁻⁵).
- Simulated quantum readout = exact centroid (±0.006 BA); logreg on all N is not better than at K=2048.
- Figure: `HPC_E13/figures/fig_e13_sstar_vs_N.png`

## Stage 4 — Under the paper's own ridge protocol, small classical sketches do lose ~0.1 accuracy (gap reproduced, not an advantage proof)
- Paper's file, its 100 random pairs, RidgeClassifier, 5-fold CV accuracy: full ridge 0.901. Classical sketches at S = 64: 0.80–0.81. Within 0.01 of full ridge needs ~1,900 genes (truncation) to ~8,192 features (hashing / sparse JL).
- Fair-baseline fixes (tuned alpha, s=4 JL, count-sketch selection) do not close the gap. The paper's baselines are general-purpose, data-oblivious methods.
- The quantum point (≈ 44 units, paper formula) assumes ideal ridge accuracy; nothing quantum was run, and sample/time costs are not compared.
- Figure: `HPC_E13/figures/fig_e13_paper_protocol.png`
