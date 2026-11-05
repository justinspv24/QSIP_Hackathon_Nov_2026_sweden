# E13 — PBMC68k (the paper's own scRNA-seq dataset)

**Summary (updated per stage)**
- Stage 1: **no memory advantage on PBMC68k.** On gene pairs (N = 1.3·10⁵ – 2.1·10⁶), a one-pass sketch matches the exact centroid with S* = 16–512 counters (S*/N ≤ 3.9·10⁻³, ≤ 4 kB). This matches PBMC3k (S* = 8–1,024).
- Stage 1: on 1,024 genes, S*/N reaches 0.5–1.0 for B and naive CD4 T. N is small and the ±0.01 band is strict; it is still ≤ 8 kB.
- Stage 1: Th2 (97 cells) has no signal (exact BA 0.55–0.60, logreg 0.50). Its S* means nothing.
- Stages 2–4: pending.

Branch `hpc-e13` (worktree `qos_cells_e13`). All numbers come from `results/*.json`, and all figures are built from that JSON.
**Not pushed**: this node has no GitHub credentials. Push with `git push -u origin hpc-e13` from the worktree.

## Data

| item | value |
|---|---|
| matrix | 10x "Fresh 68k PBMCs (Donor A)", Cell Ranger 1.1.0, hg19 filtered: `https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz`, sha256 `3f35f37f…a267931` |
| labels | Zheng et al. 2017 per-cell annotation (correlation to 11 FACS-purified populations, **not** clustering): `10XGenomics/single-cell-3prime-paper/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv`, sha256 `98dc85c0…10624b3` |
| cells / genes | 68,579 × 32,738, all cells used (no subsampling), median 525 genes / cell |
| normalisation | counts per cell → 1e4, log1p (as the PBMC3k file in E1–E7) |
| split | stratified 70/30 (split 0 = config seed 7); gene selection on train only |

Full hashes in `results/e13_data.json`. Stage 4 uses the paper's own input file (see Stage 4).

## Method (same as E7 unless noted)

- Features: genes p_g = x_g²/Σx² (top K = 1024), and gene pairs p_(g,h) ∝ x_g²x_h² (g ≤ h, top K = 512, 2048), amplitude-normalised per cell.
- Exact centroid Δ = μ₁ − μ₀ with ±1/π̂ weights; class-balanced logreg on all N.
- One-pass sketches of the record stream at M_run: count-sketch (3 rows), feature hashing, sparse JL (4 rows), Misra–Gries; oracle top-S (upper bound, not one-pass); hashing + logreg (not one-pass).
- **S\*** = smallest S (powers of 2, 8 … N) where mean BA (3 seeds) is within tol = max(0.01, one test positive) of the exact centroid **for this S and all larger S**. "Best one-pass" = min over count-sketch / hashing / sparse JL / Misra–Gries.
- Extra column **S\* (M→∞)**: the same sketch applied to the exact Δ. It separates memory from stream noise.
- d_eff = ‖Δ‖₂⁴/‖Δ‖₄⁴. Qubits = ⌈log₂N⌉ index qubits (≈ 2× logical); classical simulation only.
- Checks (`results/e13_selftest*.json`): the stream sampler has E7's law (z-rms 1.04, variance ratio 1.005). The fast pair builder equals E7's exactly. Exact centroid and logreg reproduce E7 on PBMC3k DC (0.9193 / 0.9545). The GPU Gram path equals direct pair materialisation (rel. L2 3·10⁻⁷, identical BA).
- Pairs K=2048 ran on the GPU path (E9's `pairs_gpu.py`; M→∞ and hashing+logreg with 1 seed). The CPU jobs were too slow (D11 in PROGRESS.md).

## Stage 1 — core result (`results/e13_stage1.json`, `figures/fig_e13_sstar_vs_N.png`)

| features | N | qubits | task | π̂ | test pos | exact BA | logreg BA | exact AUPRC | d_eff | best one-pass | S* | S*/N | bytes | S*/N (M→∞) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| genes K=1024 | 1,024 | 10 | B | 0.086 | 1772 | 0.848 | 0.948 | 0.792 | 20 | count-sketch | 1,024 | 1.0 | 8,232 | 0.50 |
| genes K=1024 | 1,024 | 10 | CD34+ | 0.0040 | 83 | 0.835 | 0.940 | 0.677 | 107 | Misra–Gries | 8 | 7.8e-3 | 96 | 0.031 |
| genes K=1024 | 1,024 | 10 | naive CD4 T | 0.027 | 562 | 0.742 | 0.818 | 0.062 | 88 | count-sketch | 512 | 0.50 | 4,128 | 0.25 |
| genes K=1024 | 1,024 | 10 | Th2 (no signal) | 0.0014 | 29 | 0.585 | 0.500 | 0.008 | 114 | sparse JL | 32 | 0.031 | 320 | 0.031 |
| pairs K=512 | 131,328 | 18 | B | 0.086 | 1772 | 0.849 | 0.914 | 0.794 | 1,268 | count-sketch | 512 | 3.9e-3 | 4,128 | 3.9e-3 |
| pairs K=512 | 131,328 | 18 | CD34+ | 0.0040 | 83 | 0.838 | 0.928 | 0.687 | 3,456 | sparse JL | 16 | 1.2e-4 | 192 | 1.2e-4 |
| pairs K=512 | 131,328 | 18 | naive CD4 T | 0.027 | 562 | 0.744 | 0.609 | 0.064 | 5,084 | hashing | 256 | 1.9e-3 | 2,064 | 1.9e-3 |
| pairs K=512 | 131,328 | 18 | Th2 (no signal) | 0.0014 | 29 | 0.599 | 0.500 | 0.008 | 2,232 | hashing | 256 | 1.9e-3 | 2,064 | 4.9e-4 |
| pairs K=2048 | 2,098,176 | 22 | B | 0.086 | 1772 | 0.850 | 0.902 | 0.802 | 1,453 | hashing | 512 | 2.4e-4 | 4,112 | 1.2e-4 |
| pairs K=2048 | 2,098,176 | 22 | CD34+ | 0.0040 | 83 | 0.817 | 0.946 | 0.660 | 3,868 | sparse JL | 16 | 8.0e-6 | 192 | 4.0e-6 |
| pairs K=2048 | 2,098,176 | 22 | naive CD4 T | 0.027 | 562 | 0.728 | 0.628 | 0.056 | 5,709 | sparse JL | 256 | 1.2e-4 | 2,112 | 2.4e-4 |
| pairs K=2048 | 2,098,176 | 22 | Th2 (no signal) | 0.0014 | 29 | 0.554 | 0.500 | 0.005 | 3,885 | sparse JL | 8 | 4.0e-6 | 128 | 3.9e-3 |

Counts: 68,579 cells → 48,005 train / 20,574 test for every task. Positives (train / test): B 4,136 / 1,772; CD34+ 194 / 83; naive CD4 T 1,311 / 562; Th2 68 / 29.

S* per method (— = never inside the band):

| features | task | count-sketch | hashing | sparse JL | Misra–Gries | oracle top-S | hashing+logreg |
|---|---|---|---|---|---|---|---|
| genes K=1024 | B | 1,024 | — | 1,024 | 1,024 | 128 | 128 |
| genes K=1024 | CD34+ | 64 | 64 | 128 | 8 | 8 | 16 |
| genes K=1024 | naive CD4 T | 512 | 512 | 512 | — | 64 | 128 |
| pairs K=512 | B | 512 | 1,024 | 512 | 65,536 | 4,096 | 64 |
| pairs K=512 | CD34+ | 256 | 256 | 16 | 8,192 | 8 | 8 |
| pairs K=512 | naive CD4 T | 512 | 256 | 512 | 65,536 | 8,192 | — |
| pairs K=2048 | B | 2,048 | 512 | 1,024 | 32,768 | 2,048 | 256 |
| pairs K=2048 | CD34+ | 32 | 64 | 16 | 65,536 | 8 | 8 |
| pairs K=2048 | naive CD4 T | 1,024 | 1,024 | 256 | 131,072 | 16,384 | — |

What it means:
- On gene pairs, S* stays flat (16–2,048 counters) while N grows 16×, so S*/N falls with N. PBMC3k showed the same (grey points in the figure).
- Linear sketches (hashing, JL, count-sketch) beat heavy hitters by 10–1,000× on PBMC68k. d_eff is 1,300–5,700, so Δ is spread out and a top-S list needs many entries.
- The centroid itself is a weak classifier here: B 0.85 vs logreg 0.90–0.95, and naive CD4 T AUPRC 0.06. The memory question is answered for the centroid. A stronger classifier needs more than this test shows.
- Th2: logreg overfits its 68 training positives (test BA 0.50). It is kept and flagged, not dropped.

## Limits
- One split in Stage 1 (Stage 2 adds 5 splits × 5 seeds).
- The stream is Poissonised (compound-Poisson sampler, same law as E7). Rare classes hit the record cap (Th2 runs at M/N = 218,700).
- Qubits and counters are different units. The quantum column is ⌈log₂N⌉ from a classical simulation; nothing ran on hardware.
- Annotations are from correlation to purified populations: less circular than clustering labels, but known to be noisy for T-cell subsets (Th2, naive CD4).
