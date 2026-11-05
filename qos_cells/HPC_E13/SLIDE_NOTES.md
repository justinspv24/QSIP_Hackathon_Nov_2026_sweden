# E13 backup slides (one per stage)

## Stage 1 — On PBMC68k, a few hundred classical counters match the exact centroid: no memory advantage
- Gene pairs, N = 1.3·10⁵ – 2.1·10⁶ (18–22 index qubits): best one-pass sketch S* = 16–512 counters (S*/N ≤ 3.9·10⁻³, ≤ 4 kB). PBMC3k gave 8–1,024.
- On 1,024 genes, S*/N is up to 1.0 for B (strict ±0.01 band, small N), still ≤ 8 kB. Th2 (0.14 %) has no signal (BA ≤ 0.60).
- The centroid is weak on this data (B: 0.85 vs logreg 0.90–0.95). The test covers the sketchable classifier only.
- Figure: `HPC_E13/figures/fig_e13_sstar_vs_N.png`
