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
- Best one-pass sketch S* (counters within one test cell of the exact centroid): B cells 128 → 1,024, DC 32 → 8, Mk 16 → 8 as N grows 131k → 94M. S*/N falls to 1e-5 … 8.5e-8.
- S* does not grow with N because the class signal sits in a few hundred pair features (d_eff ≈ 70–490). Limits: Louvain labels, DC 11 / Mk 5 test cells, one split.

Figure: `HPC/figures/fig_e9_sstar_vs_N.png`

---

## E10 · (filled in when E10 finishes)

Figure: `HPC/figures/fig_e10_atlas.png`

---

## E11 · Classical sketches fail only when the signal is spread over ≳ N^0.75 features — real tasks sit far below that
- Synthetic Gaussian family, N = 2¹⁴…2²⁴, signal on k = N^α features, exact-centroid balanced accuracy fixed at 0.95.
- α ≤ 0.5: best one-pass sketch (Misra–Gries) needs S* = 8–32 counters (S*/N ≤ 2e-3, falling as 1/N). α ≥ 0.75: S*/N ≥ 0.25; α = 1: S* = N.
- Real tasks (E7/E9 PBMC3k, E10 AIDA) have log d_eff / log N ≈ (filled in) — inside the "classical is enough" region. The Gaussian model is pessimistic for linear sketches (dense noise); on real data they do better.

Figure: `HPC/figures/fig_e11_advantage_map.png`
