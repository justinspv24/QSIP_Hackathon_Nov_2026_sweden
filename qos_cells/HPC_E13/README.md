# E13 — PBMC68k (the paper's own scRNA-seq dataset)

**Summary**
- Stage 1 (PBMC68k, split 0): **no memory advantage.** Gene pairs N = 1.3·10⁵–2.1·10⁶: a one-pass sketch matches the exact centroid with 16–512 counters (S*/N ≤ 3.9·10⁻³). Same as PBMC3k. 1,024 genes: S*/N up to 1.0 (small N, ≤ 8 kB).
- Stage 2 (5 splits × 5 seeds): **stable.** Pairs S*/N ≤ 0.016 on every split. PBMC68k needs 2–10× more relative memory than PBMC3k: same order.
- Stage 3 (N = 3.4·10⁷ and 1.5·10⁸, 26–28 qubits): **still no advantage.** S* = 512–4,096 counters (S*/N ≤ 3.1·10⁻⁵). S* is flat for B and naive CD4 T but grows about ∝ N for CD34+ (S*/N ≈ 2–3·10⁻⁵).
- Stage 4 (paper's ridge protocol): **the paper's gap reproduces with fair baselines.** Sketches at S = 64 give 0.80–0.81 vs 0.90 for full ridge (ideal quantum, formula only). Not an advantage claim: the quantum side is not simulated and costs are not compared.
- Th2 (97 cells) has no signal in any feature set; its S* is not informative. The centroid is weaker than logreg / ridge on this data.

Branch `hpc-e13` (worktree `qos_cells_e13`). All numbers come from `results/*.json`, and all figures are built from that JSON.
Pushed to `origin/hpc-e13` after each stage (credentials were added at 21:30; Stage 1 was pushed then).

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

## Stage 2 — robustness (`results/e13_stage2.json`, `figures/fig_e13_robustness.png`)

5 stratified 70/30 splits (split k uses seed 7 + 1000k; split 0 = Stage 1's) × 5 stream/hash seeds, for every Stage-1 task, plus PBMC3k (E7 setup: B, DC, Mk) on the same grid. 105 jobs. Pair jobs ran on the GPU path; record cap 10⁹ per seed; M→∞ variant 5 seeds; hashing+logreg 2 seeds (S ≤ 2¹⁴); no full-N logreg.

Best one-pass S*/N over the 5 splits (mean ± sd; S* never reached counts as 1):

| features | task | PBMC68k S*/N | PBMC68k S* (counters, mean) | PBMC3k task | PBMC3k S*/N |
|---|---|---|---|---|---|
| genes K=1024 | B | 0.70 ± 0.25 | 717 | B | 0.24 ± 0.15 |
| genes K=1024 | CD34+ | 0.027 ± 0.020 | 27 | DC | 0.11 ± 0.20 |
| genes K=1024 | naive CD4 T | 0.45 ± 0.10 | 461 | Mk | 0.0078 ± 0 |
| genes K=1024 | Th2 (weak, BA 0.63) | 0.28 ± 0.37 (1 split never) | — | | |
| pairs K=512 | B | 7.0e-3 ± 4.5e-3 | 922 | B | 1.7e-3 ± 1.2e-3 |
| pairs K=512 | CD34+ | 6.6e-4 ± 6.6e-4 | 86 | DC | 8.5e-5 ± 3.0e-5 |
| pairs K=512 | naive CD4 T | 2.1e-3 ± 9.6e-4 | 282 | Mk | 8.5e-5 ± 3.0e-5 |
| pairs K=512 | Th2 (weak, BA 0.63) | 9.2e-3 ± 1.1e-2 | — | | |
| pairs K=2048 | B | 4.4e-4 ± 9.8e-5 | 921 | B | 8.5e-5 ± 3.0e-5 |
| pairs K=2048 | CD34+ | 1.7e-5 ± 1.2e-5 | 36 | DC | 1.2e-5 ± 1.0e-5 |
| pairs K=2048 | naive CD4 T | 1.7e-4 ± 6.0e-5 | 359 | Mk | 1.0e-5 ± 1.0e-5 |
| pairs K=2048 | Th2 (no signal, BA 0.57) | 5.3e-5 ± 4.2e-5 | — | | |

Exact centroid BA is stable over splits (sd ≤ 0.011 on PBMC68k signal tasks; Th2 sd 0.03–0.06). Per-method S*/N for every cell is in `robustness[].methods` and printed by `scripts/e13_tables.py 2`.

What it means:
- Stage 1's verdict holds on every split. On pairs, the worst PBMC68k split needs S*/N = 0.016 (K=512 B); at K=2048 every split is ≤ 4.9·10⁻⁴ (≤ 1,024 counters).
- PBMC68k needs about 2–10× more relative memory than PBMC3k at the same N (B ~900 counters vs ~180), but it is the same order and far below N.
- On 1,024 genes the result is unstable (one split moves S*/N between 0.5 and 1.0) because N is small and the band is tight. "S* ≈ N" there means ≤ 8 kB.
- hashing+logreg shows S*/N = 1 for some tasks only because its grid stops at S = 2¹⁴ (and Th2 / naive CD4 logreg is below the centroid); it is not a true S* = N.

## Stage 3 — maximum feature space (`results/e13_stage3.json`, points in `figures/fig_e13_sstar_vs_N.png`)

Gene pairs at K = 8,192 (N = 33,558,528, 26 index qubits) and all genes detected in ≥ 3 training cells (K = 17,087–17,138 per task, N = 1.46–1.47·10⁸, 28 index qubits). Pair vectors are never built: the centroid is the upper triangle of a scaled Gram matrix of u = x²/√Z, with Z = ((Σx²)² + Σx⁴)/2, computed in row blocks on the GPU (E9's `pairs_gpu.py`, unchanged).
Check vs direct materialisation at K=512 (CD34+): identical Δ (max diff 0 in float64 output, 3·10⁻⁷ in an earlier float32 run), identical BA/AUPRC, closed-form Z within 8·10⁻⁵ (float32 input).
Timing rule: the first all-genes task (CD34+) took 25 min, so the projected time for each size was < 2 h and all 4 tasks ran (25–51 min each, one GPU per task). Peak VRAM 43.9 GB, peak RAM 35.6 GB. Logreg on all N needed scipy L-BFGS memory m=3 above N = 1.4·10⁸ (int32 workspace limit; the first attempt crashed, D16).

| features | N | qubits | task | exact BA | logreg BA (all N) | quantum-sim BA | d_eff | best one-pass | S* | S*/N | bytes | S*/N (M→∞) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pairs K=8192 | 3.36·10⁷ | 26 | B | 0.848 | 0.889 | 0.848 | 1,564 | hashing | 1,024 | 3.1e-5 | 8,208 | 1.5e-5 |
| pairs K=8192 | 3.36·10⁷ | 26 | CD34+ | 0.843 | 0.940 | 0.843 | 3,551 | count-sketch | 1,024 | 3.1e-5 | 8,232 | 1.2e-4 |
| pairs K=8192 | 3.36·10⁷ | 26 | naive CD4 T | 0.718 | 0.596 | 0.718 | 5,811 | hashing | 512 | 1.5e-5 | 4,112 | 1.5e-5 |
| pairs K=8192 | 3.36·10⁷ | 26 | Th2 (no signal) | 0.559 | 0.500 | 0.558 | 7,368 | hashing | 8 | 2.4e-7 | 80 | 9.8e-4 |
| pairs, all genes | 1.46·10⁸ | 28 | B | 0.846 | 0.869 | 0.846 | 1,569 | hashing | 1,024 | 7.0e-6 | 8,208 | 7.0e-6 |
| pairs, all genes | 1.47·10⁸ | 28 | CD34+ | 0.847 | 0.946 | 0.841 | 3,440 | count-sketch | 4,096 | 2.8e-5 | 32,808 | 7.0e-6 |
| pairs, all genes | 1.46·10⁸ | 28 | naive CD4 T | 0.711 | 0.602 | 0.714 | 5,819 | hashing | 512 | 4.0e-6 | 4,112 | 1.0e-6 |
| pairs, all genes | 1.46·10⁸ | 28 | Th2 (no signal) | 0.578 | 0.500 | 0.566 | 9,815 | count-sketch | 524,288 | 3.6e-3 | 4.2 MB | 4.4e-7 |

What it means:
- Up to N = 1.5·10⁸ the best one-pass sketch needs ≤ 4,096 counters (≤ 33 kB) for every task with signal. The verdict of Stages 1–2 extends to the largest space.
- S* is not always flat. For B and naive CD4 T it stays at 512–1,024 from N = 1.3·10⁵ to 1.5·10⁸. For CD34+ it grows from ~36 (K=2048, Stage-2 mean) to 1,024 and 4,096, i.e. roughly ∝ N with S*/N ≈ 2–3·10⁻⁵. If that trend continued, a sketch would need ~3·10⁻⁵·N counters.
- The simulated quantum readout (p · sin(φ), phases rescaled) gives the same BA as the exact centroid within 0.006, and the stream converges at M/N = 900–2,700 (72,900 for Th2).
- Logreg on all N is below the centroid for naive CD4 T (0.60 vs 0.71): it overfits; more N does not help it.

## Stage 4 — the paper's protocol (`results/e13_stage4.json`, `figures/fig_e13_paper_protocol.png`)

Setup copied from the paper's code (`quantum-oracle-sketching/real_datasets/pbmc68k_svm.py`, `sketch_utils.py`, commit 07e9167):

| item | value |
|---|---|
| data | the paper's own input: scvelo PBMC68k h5ad (`https://ndownloader.figshare.com/files/27686886`, sha256 `c93b1cca…dfa26fb`), 65,877 cells × 33,939 genes, same Zheng labels; log1p(raw counts), genes with ≥ 1 count → D = 16,122 |
| pairs | the paper's 100 random draws (rng 42, with replacement) = 46 unique pairs; each evaluated once, means weighted by draw count |
| model | RidgeClassifier(alpha=200, class_weight='balanced'), 5-fold CV **accuracy** (the paper's metric; balanced accuracy also stored) |
| classical, paper | its own code: balanced unsigned feature hashing, balanced signed sparse JL (s=1), rare-gene truncation; 5 sketch seeds |
| classical, added | sparse JL s=4; ridge with alpha tuned by nested leave-one-out; count-sketch gene selection (S counters) + ridge; one-pass count-sketch centroid |
| quantum | the paper's machine-size formula, 2⌈log₂(n+2D)⌉ + ⌈log₂(s+1)⌉ + 4 = 42–44 (mean 44); log₂D = 14 index qubits (≈ 28 logical). **Formula only; nothing quantum was simulated or run.** Its ideal accuracy = ridge on all genes |

Reference: ridge on all genes = **0.901 ± 0.015**; exact centroid = 0.845.

5-fold CV accuracy vs classical machine size S:

| method | S=16 | 64 | 256 | 1,024 | 4,096 | S within 0.01 of full ridge |
|---|---|---|---|---|---|---|
| paper: sparse JL s=1 + ridge | 0.731 | 0.809 | 0.852 | 0.872 | 0.886 | 8,192 |
| paper: feature hashing + ridge | 0.732 | 0.803 | 0.848 | 0.871 | 0.886 | 8,192 |
| sparse JL s=4 + ridge | 0.721 | 0.806 | 0.852 | 0.871 | 0.886 | 8,192 |
| sparse JL s=1 + ridge, alpha tuned | 0.727 | 0.807 | 0.856 | 0.878 | (≤ 2,048 run) | — |
| count-sketch gene selection + ridge | 0.461 | 0.682 | 0.828 | 0.870 | 0.894 | 4,096 |
| one-pass count-sketch centroid | 0.691 | 0.786 | 0.833 | 0.845 | 0.844 | never (centroid ceiling 0.845) |

Paper's truncation (keep the most-detected genes): 74 genes 0.811, 157 → 0.856, 532 → 0.878, 1,934 → 0.892 (within 0.01 of full ridge).

What it means:
- **Under the paper's protocol, the gap it reports is reproduced.** At S ≈ 44–64 (the paper's quantum machine size) every classical method tested reaches 0.79–0.81. The ideal quantum LS-SVM would give 0.90. Matching full ridge takes ~2,000 (truncation) to ~8,000 (sketches) classical numbers.
- The fair-baseline fixes do not close it. Tuning alpha changes accuracy by < 0.01. Sparse JL s=4 equals s=1. Count-sketch selection is collision-dominated at S ≤ 32 (≈ 3 buckets per row: decoded |Δ| ties) and only catches up at S ≥ 1,024.
- This is a different classifier from Stages 1–3. There the quantum sketch computes the class centroid, and a few hundred classical counters match it. Here the target is ridge / LS-SVM, which needs the inverse covariance, and oblivious sketches lose accuracy slowly.
- **Not a demonstrated quantum advantage.** The quantum point is a formula with an assumed ideal accuracy. Sample count, run time, QSVT error and readout shots are not in the comparison. A qubit and a float are different units. Truncation needs D counters in its counting pass, which the paper's "machine size" does not count.
- **Note: the paper's classical baselines are general-purpose, data-oblivious methods** (feature hashing, sparse JL, truncation). They are not streaming learners designed for ridge (e.g. frequent-directions or sketched-covariance ridge). Those were not tested here, so the classical side is not shown to be optimal.

## Limits
- One split in Stage 1 (Stage 2 adds 5 splits × 5 seeds).
- The stream is Poissonised (compound-Poisson sampler, same law as E7). Rare classes hit the record cap (Th2 runs at M/N = 218,700).
- Qubits and counters are different units. The quantum column is ⌈log₂N⌉ from a classical simulation; nothing ran on hardware.
- Annotations are from correlation to purified populations: less circular than clustering labels, but known to be noisy for T-cell subsets (Th2, naive CD4).
