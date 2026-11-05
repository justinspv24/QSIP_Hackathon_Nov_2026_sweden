# SLIDE NOTES — backup slides E8–E11

One slide per experiment. Title = the verdict. The quantum column everywhere is a classical simulation of the sketch.

---

## E8 · The GPU circuit gives the same numbers as the sketch formula, up to 30 index qubits — a correctness check, not an advantage
- Hadamard-test sketch circuit (H layer, 4.2 M commuting controlled-phase gates, S†, H) simulated gate by gate on one A6000: |Z_circuit − Z_formula| ≤ 3.1e-16 for n = 10…30 index qubits (+1 ancilla).
- n = 30 (2³¹ amplitudes, complex128): 31.5 s, 40.8 GB VRAM; shot noise follows 1/√shots (98.5–100 % of repeats inside 3σ).
- The circuit is diagonal and classically easy; this shows the simulator and formula agree at scale, nothing more.

Figure: `HPC/figures/fig_e8_scaling.png`

---

## E9 · On PBMC3k, classical sketches need only 8–1,024 counters, even at N = 9.4·10⁷ gene pairs: no memory advantage
- Full gene-pair space up to all 13,714 genes (N = 94,043,755, 27 index qubits), computed in K×K form on GPU; gate at K = 512 reproduces E7 (exact centroid and logreg identical).
- Best one-pass sketch S* (counters within one test cell of the exact centroid): B cells 128 → 1,024, DC 32 → 8, Mk 16 → 8 as N grows 131k → 94M. S*/N falls to 1e-5 … 8.5e-8. Logreg on all 94M pairs: 0.994 / 0.955 / 0.800, the same as at K = 512.
- S* does not grow with N because the class signal sits in a few hundred pair features (d_eff ≈ 70–490). Limits: Louvain labels, DC 11 / Mk 5 test cells, one split.

Figure: `HPC/figures/fig_e9_sstar_vs_N.png`

---

## E10 · On a 1.27-million-cell atlas with curated labels, 16–512 counters still solve every rare-cell task: rare classes cost records, not memory
- AIDA (CELLxGENE Census 2025-11-08): 1,265,624 PBMCs, 625 donors, expert labels, donor split, no subsampling. Tasks: HSC (π 1.1e-4, 32 test cells from 19 unseen donors), plasma cell (1.3e-3), CD1c+ DC (1.0e-2), CD14+ monocyte (0.16).
- Gene pairs at N = 131,328 and 2.1 M: the best one-pass sketch matches the exact centroid (bal. acc. 0.90–0.99) with 16–512 counters (S*/N = 3e-5 … 4e-3). Same verdict as PBMC3k, now without circular labels or tiny test sets.
- Records needed for 10 % error follow the closed form within 7 % and scale as 1/(π(1−π)) over a 1,400× range of π (HSC, K = 2,048: 9.2·10¹² records), for quantum and classical sketches alike.

Figure: `HPC/figures/fig_e10_atlas.png`

---

## E11 · Classical sketches fail only when the signal is spread over ≳ N^0.75 features, and no real task is there
- Synthetic Gaussian family, N = 2¹⁴…2²⁴, signal on k = N^α features, exact-centroid balanced accuracy fixed at 0.95.
- α ≤ 0.5: the best one-pass sketch (Misra–Gries) needs 8–32 counters (S*/N ≤ 2e-3, falling as 1/N). α ≥ 0.75: S*/N ≥ 0.25; α = 1: S* = N.
- 20 real gene-pair tasks (PBMC3k K = 512…13,714, AIDA K = 512/2,048): log d_eff / log N = 0.23–0.74, measured S*/N ≤ 4e-3. AIDA HSC sits at 0.74, next to the synthetic edge, yet needs 16 counters, so the isotropic-noise model is pessimistic and d_eff alone does not predict failure.

Figure: `HPC/figures/fig_e11_advantage_map.png`
