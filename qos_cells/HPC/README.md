# HPC scale-up: E8–E11

**Summary**
- **Verdict:** at every scale we tried, small one-pass classical sketches match the exact class centroid, so there is **no memory advantage** for the quantum oracle sketch on this data. The quantum column is a classical simulation, and no quantum advantage is claimed.
- **E8:** the GPU statevector circuit equals the sketch formula to **3.1·10⁻¹⁶** up to **30 index qubits** (+1 ancilla). This checks correctness and scaling only; the circuit is diagonal and classically easy.
- **E9:** on PBMC3k gene pairs up to **N = 9.4·10⁷** (27 qubits), the best one-pass sketch needs **S\* = 8–1,024 counters** (S\*/N down to 8.5·10⁻⁸). S\* does not grow with N.
- **E10:** AIDA atlas (**1.27 M cells, 625 donors, curated labels**, donor split), 4 tasks with π from 0.16 to **1.2·10⁻⁴**. Best one-pass S\* is **16–512 counters** of N = 131,328 (K = 512) and 2.1 M (K = 2048). The rare-class cost is in **records**: measured M₁₀ matches the closed form (ratio **0.93–1.04**) and follows the 1/(π(1−π)) law.
- **E11:** in a synthetic Gaussian model, sketches fail once the signal spreads over ≳ N^0.75 features. Real tasks have log d_eff / log N = **0.23–0.74**, yet are still solved with S\*/N ≤ **4·10⁻³**. That Gaussian model is pessimistic.

All numbers are in `results/*.json`, and every figure is built from those files only. Node: quant1 (8× RTX A6000 48 GB, 128 cores, 2 TB RAM); see `ENV.md`. Run log and decisions: `PROGRESS.md`.

## Experiments at a glance

| exp | question | status | wall time (compute) | peak RAM / VRAM | verdict |
|---|---|---|---|---|---|
| E8 | circuit = formula at scale? | done | 50 s | 9.6 GB / 40.8 GB | yes, to 3.1e-16 up to n = 30; correctness check only |
| E9 | S\* vs N up to all genes | done | K=13714: 6.6 min (+ 5.8 min logreg) | 21 GB / 41 GB | S\* flat (8–1,024) while N grows 700×; no memory advantage |
| E10 | curated labels, big test sets | done | 72 min incl. 12 min download + 12 min prep; longest run 23 min | 188 GB (prep) / 41 GB | 16–512 counters suffice for every task, π down to 1.1e-4; rare classes cost records (1/(π(1−π)) law confirmed), not memory |
| E11 | where do sketches fail? | done | 55 s | 0.4 GB / 5.8 GB | only when the signal spreads over ≳ N^0.75 features (synthetic); real tasks are solved anyway |

## How to rerun

```bash
cd qos_cells
bash HPC/run_all.sh          # creates HPC/.venv from requirements.txt, downloads AIDA (14 GB), runs everything, ~45 min on quant1
# single pieces
HPC/.venv/bin/python HPC/scripts/e9_fullpairs.py run --K 8192 --seeds 5 --part main
HPC/.venv/bin/python HPC/scripts/e9_fullpairs.py merge
HPC/.venv/bin/python HPC/scripts/plots.py            # all figures from JSON
```

- **Code:** `scripts/pairs_gpu.py` is the shared GPU library: pair space in K×K form, the stream sampler, the sketches, and logistic regression. `scripts/hpc_common.py` handles metadata, JSON, and peak RAM/VRAM tracking. Existing repo scripts are imported, not edited. E7's S\* rules (`_finalise`, `_verdict`, `summarise`) come from `scripts/09_route2_classical_memory.py`.
- **Checks:** `scripts/tests/check_pairs.py` compares the GPU library with the E7 code at K = 512:

  | quantity | difference from E7 |
  |---|---|
  | pair norm | 1e-13 |
  | pair entries | 8e-10 |
  | centroid | 1e-6 |
  | scores | 1e-7 |
  | hashes | identical |

  The stream error matches the closed form.
- **Data:** raw data, the venv and caches are gitignored (`HPC/data/`, `HPC/.venv/`). No committed file is over 50 MB.

## How the gene-pair space is handled

The brief says never to build per-cell pair vectors. Let u = x²/√Z with Z = ‖pair(x)‖² = ((Σx²)² + Σx⁴)/2. Then the pair feature is p_(g,h) = u_g·u_h. Every quantity we need becomes a K×K operation:

| quantity | K×K form |
|---|---|
| class centroid | triu(Uᵀ U / n), a Gram matrix |
| score of any pair-space weight D | uᵀ T u, with T the upper-triangular matrix holding D |
| logistic-regression gradient | triu(Uᵀ diag(r) U) |

The E7 Poissonised stream (Poisson(λ) records per (cell, pair) entry) is drawn by Poisson splitting. The total number of records is Poisson(λE), and each record lands on a uniformly random nonzero entry: pick a cell with weight m(m+1)/2, then a uniform pair within it. The distribution is the same as E7's and is never materialised; the sampler runs at ~3·10⁸ records/s per GPU.

---

## E8 · GPU circuit check

- **Circuit:** H on every index qubit, then a Hadamard-test ancilla, then M = 4,194,304 controlled diagonal phase gates exp(iτw/M·|g⟩⟨g|) from a synthetic stream (±1/π weights, π = 0.02), then S† and H on the ancilla.
- **Simulator:** CuPy complex128 statevector with in-place gate kernels, applied gate by gate. Gates that share a target index are applied in rounds; this is exact because the gates commute.
- **Test states:** "uniform" is built from the H gates. "cell" has 4,096 sparse amplitudes and is set directly in the statevector, not decomposed into gates.

| n index qubits | amplitudes | max \|Z_circuit − Z_formula\| | max phase deviation | wall s | peak VRAM GB |
|---|---|---|---|---|---|
| 10 | 2^11 | 1.1e-16 | 6.9e-15 | 7.3 | 1.0 |
| 14 | 2^15 | 6.8e-17 | 6.9e-15 | 2.7 | 0.8 |
| 18 | 2^19 | 3.8e-17 | 3.8e-15 | 1.2 | 0.7 |
| 22 | 2^23 | 1.2e-16 | 2.0e-15 | 0.9 | 0.8 |
| 26 | 2^27 | 3.6e-17 | 5.6e-16 | 2.4 | 3.3 |
| 30 | 2^31 | 3.1e-16 | 5.8e-16 | 31.5 | 40.8 |

**Shot noise:** for 10²…10⁶ shots with 200 repeats each, the RMS error follows 1/√shots, and 98.5–100 % of repeats fall within 3σ.

**Caveat:** the sketch circuit is diagonal (an H layer, commuting phase gates and a Hadamard test), so it is classically easy to simulate. E8 is a correctness and scaling check of simulator against formula. It is not evidence of quantum advantage. Two further limits: the records are synthetic, not PBMC data, and the qiskit cross-check was not run.

Figure: `figures/fig_e8_scaling.png` · JSON: `results/e8_gpu_circuit.json`

---

## E9 · Full gene-pair space on PBMC3k

**Setup:** same task, split (seed 7), gene rule, stream, methods and S\* rules as E7. "Best one-pass" is E7's verdict rule: the stable S\* within max(0.01, one test cell) of the exact centroid, excluding hashing + logreg (not one-pass) and oracle top-S (a bound).

**Gate at K = 512 (rerun of E7 with the new code):**

| class | exact BA new / E7 | logreg new / E7 | stream new / E7 | best one-pass S\*/N new / E7 | max grid steps apart (any method) |
|---|---|---|---|---|---|
| B cells | 0.9590 / 0.9590 | 0.9944 / 0.9944 | 0.963 / 0.959 | 9.7e-04 (hashing) / 9.7e-04 (hashing) | 1 |
| Dendritic cells | 0.9193 / 0.9193 | 0.9545 / 0.9545 | 0.916 / 0.911 | 2.4e-04 (Misra–Gries) / 2.4e-04 (hashing) | 3 |
| Megakaryocytes | 0.8962 / 0.8962 | 0.8000 / 0.8000 | 0.895 / 0.874 | 1.2e-04 (hashing) / 6.1e-05 (sparse JL) | 1 |

Deterministic quantities are identical. Stochastic S\* values differ by stream random numbers; the largest gap is DC, where one test cell moves balanced accuracy by 0.045. **The gate passes.**

**Results:**

| K | N | qubits | class | test + | exact BA | logreg BA (all N) | stream BA | quantum-sim BA | M/N | best one-pass (S\*) | S\*/N | d_eff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 131,328 | 18 | B cells | 103 | 0.959 | 0.994 | 0.963 | 0.965 | 100 | hashing (128) | 9.7e-04 | 365 |
| 512 | 131,328 | 18 | Dendritic cells | 11 | 0.919 | 0.955 | 0.916 | 0.915 | 100 | Misra–Gries (32) | 2.4e-04 | 415 |
| 512 | 131,328 | 18 | Megakaryocytes | 5 | 0.896 | 0.800 | 0.895 | 0.890 | 8100 | hashing (16) | 1.2e-04 | 91 |
| 2,048 | 2,098,176 | 22 | B cells | 103 | 0.948 | 0.994 | 0.950 | 0.951 | 100 | count-sketch (256) | 1.2e-04 | 401 |
| 2,048 | 2,098,176 | 22 | Dendritic cells | 11 | 0.921 | 0.955 | 0.924 | 0.923 | 100 | Misra–Gries (32) | 1.5e-05 | 460 |
| 2,048 | 2,098,176 | 22 | Megakaryocytes | 5 | 0.796 | 0.800 | 0.854 | 0.830 | 100 | hashing (8) | 3.8e-06 | 85 |
| 8,192 | 33,558,528 | 26 | B cells | 103 | 0.947 | 0.985 | 0.947 | 0.947 | 100 | count-sketch (512) | 1.5e-05 | 405 |
| 8,192 | 33,558,528 | 26 | Dendritic cells | 11 | 0.892 | 0.955 | 0.898 | 0.898 | 100 | Misra–Gries (16) | 4.8e-07 | 483 |
| 8,192 | 33,558,528 | 26 | Megakaryocytes | 5 | 0.797 | 0.800 | 0.841 | 0.858 | 100 | Misra–Gries (8) | 2.4e-07 | 68 |
| 13,714 | 94,043,755 | 27 | B cells | 103 | 0.948 | 0.994 | 0.947 | 0.946 | 100 | count-sketch (1,024) | 1.1e-05 | 407 |
| 13,714 | 94,043,755 | 27 | Dendritic cells | 11 | 0.894 | 0.955 | 0.897 | 0.895 | 100 | Misra–Gries (8) | 8.5e-08 | 489 |
| 13,714 | 94,043,755 | 27 | Megakaryocytes | 5 | 0.797 | 0.800 | 0.863 | 0.862 | 300 | count-sketch (8) | 8.5e-08 | 67 |

- **S\* stays flat.** N grows 716× (2¹⁷ → 2²⁷), while the best one-pass S\* stays at 8–1,024. The class signal lives in a few hundred pair features: d_eff = ‖Δ‖₂⁴/‖Δ‖₄⁴ ≈ 70–490, and it does not grow with K.
- **Reservoir is the exception.** It needs S ≈ N for B cells.
- **Mk is noisy.** It has 5 test cells, and one cell moves balanced accuracy by 0.1. At K ≥ 2048 the stream estimate even beats the exact centroid (0.85 vs 0.80), so treat the Mk numbers as noise-level.
- **Scope:**
  - Seeds: 5 at K ≤ 8,192, 3 at K = 13,714.
  - Reservoir dropped at K = 13,714.
  - Hashing + logreg: S ≤ 2¹⁶, 3 seeds.
  - The all-genes point fit easily (6.6 min), so nothing was cut.
- **Logistic regression on all N:** class-balanced, C = 1, solved in K×K form. At N = 9.4·10⁷ it uses L-BFGS history 5 instead of 10 (scipy's int32 workspace overflows otherwise); the optimum is the same.

Figure: `figures/fig_e9_sstar_vs_N.png` · JSON: `results/e9_fullpairs.json` (merged), `results/e9_part_K*_*.json` (per run)

---

## E10 · AIDA atlas with curated labels

**Dataset**
- **Source:** AIDA Phase 1 Data Freeze v2 (Asian Immune Diversity Atlas), blood PBMCs, 10x 5′ v2.
- **Version:** CELLxGENE Census 2025-11-08, dataset `c838aec3-03ef-4398-b882-0e3912abfff0`, source h5ad 14,244,724,636 B, sha256 `a5bce74985df8cbe8cf1f18877a1607ed556d99061dc08e7bdbc4dbae732c593`.
- **Size:** 1,265,624 cells, 35,477 genes, 625 donors.
- **Labels:** `obs.cell_type`, the authors' expert annotation mapped to the Cell Ontology. We did no re-clustering.
- **Preprocessing:** raw counts, normalised to 10⁴ per cell, then log1p (same as PBMC3k `.raw`).
- **Split:** by donor, with 30 % of donors in test (GroupShuffleSplit, seed 7): train 892,135 cells from 437 donors, test 373,489 cells from 188 donors.
- **No subsampling:** every cell is used, so each π is the true atlas prevalence.
- **Pipeline:** gene selection is train-only (E7 rule), followed by the E7 pipeline for genes (N = 512) and pairs (K = 512 → N = 131,328, and K = 2,048 → N = 2,098,176), with 3 stream/hash seeds.

**Tasks (one-vs-rest)**

| task | cells in atlas | π | train + / test + | donors with positives (train/test) |
|---|---|---|---|---|
| hematopoietic stem cell | 135 | 1.1e-4 | 103 / 32 | 76 / 19 |
| plasma cell | 1,672 | 1.3e-3 | 1,252 / 420 | 321 / 138 |
| CD1c+ myeloid DC | 13,207 | 1.0e-2 | 9,232 / 3,975 | 410 / 181 |
| CD14+ monocyte (common) | 202,124 | 0.16 | 142,654 / 59,470 | 414 / 181 |

**Results** (balanced accuracy on test; "best one-pass" = E7 verdict rule)

| task | space | N | exact | logreg (all N) | stream (M/N) | quantum-sim | best one-pass (S\*) | S\*/N | d_eff |
|---|---|---|---|---|---|---|---|---|---|
| HSC | genes | 512 | 0.951 | 0.969 | 0.944 (218,700) | 0.957 | hashing (256) | 5.0e-01 | 95 |
| HSC | pairs K=512 | 131,328 | 0.945 | 0.859 | 0.951 (218,700) | 0.952 | sparse JL (16) | 1.2e-04 | 6,091 |
| HSC | pairs K=2048 | 2,098,176 | 0.904 | 0.828 | 0.907 (218,700) | 0.907 | count-sketch (64) | 3.1e-05 | 7,933 |
| plasma cell | genes | 512 | 0.988 | 0.983 | 0.988 (24,300) | 0.988 | count-sketch (64) | 1.2e-01 | 25 |
| plasma cell | pairs K=512 | 131,328 | 0.988 | 0.981 | 0.988 (24,300) | 0.988 | hashing (64) | 4.9e-04 | 1,131 |
| plasma cell | pairs K=2048 | 2,098,176 | 0.988 | 0.986 | 0.987 (24,300) | 0.987 | hashing (128) | 6.1e-05 | 937 |
| CD1c+ DC | genes | 512 | 0.956 | 0.991 | 0.956 (2,700) | 0.951 | hashing (256) | 5.0e-01 | 37 |
| CD1c+ DC | pairs K=512 | 131,328 | 0.957 | 0.972 | 0.956 (900) | 0.955 | count-sketch (512) | 3.9e-03 | 2,136 |
| CD1c+ DC | pairs K=2048 | 2,098,176 | 0.969 | 0.941 | 0.970 (2,700) | 0.969 | hashing (512) | 2.4e-04 | 2,974 |
| CD14+ mono | genes | 512 | 0.961 | 0.986 | 0.960 (100) | 0.958 | hashing (16) | 3.1e-02 | 63 |
| CD14+ mono | pairs K=512 | 131,328 | 0.962 | 0.987 | 0.963 (100) | 0.962 | hashing (32) | 2.4e-04 | 4,466 |
| CD14+ mono | pairs K=2048 | 2,098,176 | 0.963 | 0.987 | 0.963 (100) | 0.962 | sparse JL (64) | 3.1e-05 | 6,351 |

AUPRC for every row is in the JSON.

**Rare-class sample cost**

M₁₀ is the number of records needed for 10 % relative L2 error of the streamed Δ. The measured value comes from a c/√M fit over the M grid, mean of 3 seeds. The closed form is M₁₀ = 100·N·(q₁/π + q₀/(1−π))/‖Δ‖², which is the 1/(π(1−π)) law times a task constant.

| task | space | π | M₁₀ measured | M₁₀ closed form | ratio | fitted slope | M₁₀·π(1−π)/N |
|---|---|---|---|---|---|---|---|
| HSC | genes | 1.2e-04 | 3.98e+09 | 3.80e+09 | 1.04 | −0.50 | 896 |
| HSC | pairs K=512 | 1.2e-04 | 5.93e+11 | 5.90e+11 | 1.01 | −0.50 | 522 |
| HSC | pairs K=2048 | 1.2e-04 | 9.20e+12 | 9.20e+12 | 1.00 | −0.50 | 506 |
| plasma cell | genes | 1.4e-03 | 1.96e+08 | 1.92e+08 | 1.02 | −0.48 | 536 |
| plasma cell | pairs K=512 | 1.4e-03 | 5.64e+10 | 5.80e+10 | 0.97 | −0.51 | 602 |
| plasma cell | pairs K=2048 | 1.4e-03 | 1.27e+12 | 1.26e+12 | 1.01 | −0.52 | 850 |
| CD1c+ DC | genes | 1.0e-02 | 2.49e+07 | 2.44e+07 | 1.02 | −0.51 | 498 |
| CD1c+ DC | pairs K=512 | 1.0e-02 | 3.28e+09 | 3.29e+09 | 1.00 | −0.50 | 256 |
| CD1c+ DC | pairs K=2048 | 1.0e-02 | 7.70e+10 | 7.73e+10 | 1.00 | −0.50 | 376 |
| CD14+ mono | genes | 1.6e-01 | 1.19e+06 | 1.28e+06 | 0.93 | −0.50 | 313 |
| CD14+ mono | pairs K=512 | 1.6e-01 | 1.86e+08 | 1.86e+08 | 1.00 | −0.50 | 190 |
| CD14+ mono | pairs K=2048 | 1.6e-01 | 3.81e+09 | 3.79e+09 | 1.01 | −0.50 | 244 |

**Reading**
- **Sketches:** in every task and both pair spaces, a one-pass classical sketch with **16–512 counters** reaches the exact centroid (S\*/N = 3·10⁻⁵ … 4·10⁻³). The same holds for the rarest type (HSC, π = 1.1·10⁻⁴, 32 test cells from 19 unseen donors). The curated labels and large test sets do not change the E7 verdict.
- **Sample cost:** the rare-class cost is in **records**, as predicted. Measured M₁₀ equals the closed form within 7 %, with a slope of −0.50. M₁₀·π(1−π)/N stays at 190–900 while π varies 1,400×. For HSC with gene pairs at K = 2,048 that is 9.2·10¹² records, which the quantum and classical sketches pay alike.
- **K = 2,048 does not help:** it is not better than K = 512 except for CD1c+ DC (0.969 vs 0.957), and HSC drops (0.904 vs 0.945).
- **Logreg is not a reliable baseline here:** class-balanced logistic regression on all N (C = 1, untuned, as in E7) is below the centroid for HSC pairs (0.86 / 0.83; 103 positives in 1.3·10⁵–2·10⁶ features) and for CD1c+ DC at K = 2,048 (0.94; hit 300 iterations). It is a reference, not a tuned competitor.
- **Gene-only space (N = 512):** S\*/N is large there only because N is tiny (S\* = 16–256 counters).

**Cuts and decisions**
- **hashing + logreg:** at K = 512 only, S ∈ {64, 256, 1024}, 1 seed; it is off at K = 2,048 (it would hash ~10¹¹ pair entries per seed).
- **Records cap:** first 4·10⁹ per seed, then rerun with 10¹¹ for the three runs that hit it (D17). Superseded results are in `logs/superseded/`.
- **Convergence rule:** E7's rule plus "rel. error < 1" (D14).

Figure: `figures/fig_e10_atlas.png` · JSON: `results/e10_atlas.json` (merged), `results/e10_part_*.json` (per run)

---

## E11 · Advantage map

**Synthetic family:**
- N = 2¹⁴…2²⁴, with the signal on k = N^α random features (equal size, random signs), α ∈ {0, 0.25, 0.5, 0.75, 1}.
- Training noise: Δ̂ = Δ + τg, with τ² = 1/26 + 1/1820 (PBMC3k DC sizes).
- SNR fixed so that the exact-centroid balanced accuracy is 0.95.
- Test cells are x = ±Δ/2 + N(0, I), so the balanced accuracy of any decoded D is exactly Φ(⟨Δ,D⟩ / 2‖D‖).
- 3 seeds.
- S\* = best of count-sketch, hashing, sparse JL and Misra–Gries (stable rule).

| α \ log₂N | 14 | 16 | 18 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|
| 0 | 4.9e-4 | 1.2e-4 | 3.1e-5 | 7.6e-6 | 1.9e-6 | 4.8e-7 |
| 0.25 | 4.9e-4 | 1.2e-4 | 3.1e-5 | 7.6e-6 | 1.9e-6 | 4.8e-7 |
| 0.5 | 2.0e-3 | 4.9e-4 | 1.2e-4 | 3.1e-5 | 7.6e-6 | 1.9e-6 |
| 0.75 | 0.25 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| 1 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

(best one-pass S\*/N; every N value is in the JSON)

- **Who wins:** Misra–Gries is best in every cell. For α ≤ 0.5 it needs S\* = 8–32 counters.
- **Where it breaks:** once k > √N, the per-feature signal falls below the training noise, and every sketch needs S ≥ N/4.
- **Linear sketches never reach the target here:** count-sketch, hashing and sparse JL fail at every S ≤ N, because isotropic noise on all N features collides into the signal buckets. On real data (E7, E9, E10) these same sketches win, so this model is **pessimistic** for them.
- **Real tasks:** d_eff of the exact centroid for every gene-pair task in E7/E9 (PBMC3k, K = 512…13,714) and E10 (AIDA, K = 512 and 2,048) gives y = **0.23–0.74**, and measured best S\*/N is **8.5·10⁻⁸ … 3.9·10⁻³**. The most spread task (AIDA HSC, pairs K = 512, y = 0.74) sits next to the synthetic failure edge (y ≈ 0.75) but needs only 16 counters (S\*/N = 1.2·10⁻⁴). So d_eff alone does not predict where classical sketches fail. No real task falls in the failure region.
- **Not on the map:** gene-only spaces (N = 512) are outside it.

Figure: `figures/fig_e11_advantage_map.png` · JSON: `results/e11_advantage_map.json` (`synthetic`, `real_points`, `caption`; the GPU run's resources are in `run_resources`)

---

## Limits (all experiments)
- **Simulation only:** the quantum column is a classical simulation of the sketch (p · sin(cφ)). Nothing ran on quantum hardware, and no quantum advantage is claimed.
- **Units:** S counts classical counters. A qubit and a float are not the same unit, and S\* is not compared one-to-one with log₂N.
- **PBMC3k (E9):** Louvain labels are circular, rare test sets are tiny (DC 11, Mk 5), and there is one split.
- **AIDA (E10):** one atlas and one donor split. The labels are curated, but the atlas authors still derived them from the same expression data plus expert review.
- **Stream:** Poissonised, drawn by Poisson splitting rather than record by record. Its first and second moments equal E7's.
- **Git:** pushing failed because this node has no GitHub credentials. All commits are local on branch `hpc` and must be pushed from a machine with access.
