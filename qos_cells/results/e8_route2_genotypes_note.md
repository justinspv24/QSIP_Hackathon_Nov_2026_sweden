## E8: Route 2 on a dense-signal dataset (1000 Genomes genotypes)

**Why.** PBMC3k signal is sparse (marker genes), so classical sketches won (E7). Close human populations differ by small allele-frequency shifts spread over many SNPs (density screen: `results/e7b_g1k_density_screen.json`). Data: 1000 Genomes phase 3 (20130502), EUR + EAS, chromosomes 17,18,19,20,21,22; 791,251 SNPs (biallelic SNP, MAF >= 0.05, no missing call, over all 1,007 EUR+EAS people (label-free)); dosage 0/1/2, amplitude p = x^2/||x||^2, no supervised selection. Same stream, weights, seeds and S* rule as E7, plus a **random SNP subset** baseline (S exact counters on S random SNPs), which is the fair classical baseline when the signal is dense but redundant. Scaling test: nested random SNP subsets N/8 → N. Script `scripts/10_route2_genotypes.py`, figure `figures/fig_e8_genotype_memory.png`.

**Best one-pass classical S* (stable, tol = max(0.01, one test positive)) vs N, and log–log slope** (1 = memory grows like N, 0 = flat):

| task | exact centroid at full N | N = 98,906 | N = 197,813 | N = 395,626 | N = 791,251 | slope |
|---|---|---|---|---|---|---|
| JPT vs CHB (104 vs 103) | 1.00 / AUPRC 1.00 | 8,192 (0.083, hashing) | 8,192 (0.041, sparse JL) | 8,192 (0.021, sparse JL) | 8,192 (0.01, count-sketch) | 0.00 |
| JPT vs rest of East Asia (104 vs 400) | 0.98 / AUPRC 1.00 | 98,906 (1, sparse JL) | 16,384 (0.083, hashing) | 32,768 (0.083, count-sketch) | 16,384 (0.021, sparse JL) | -0.68 |
| FIN vs rest of Europe (99 vs 404) | 0.97 / AUPRC 1.00 | 8,192 (0.083, hashing) | 8,192 (0.041, hashing) | 16,384 (0.041, hashing) | 16,384 (0.021, hashing) | 0.40 |

**Per method at full N** (stable S*, S*/N in brackets; quantum = classical simulation of the sketch, same records):

| task | count-sketch | hashing | sparse JL | Misra–Gries | reservoir | random SNP subset | oracle top-S | hashing+logreg | sketch p·sin(cφ) | logreg, all N |
|---|---|---|---|---|---|---|---|---|---|---|
| JPT vs CHB | 8,192 (0.01) | 16,384 (0.021) | 16,384 (0.021) | 65,536 (0.083) | 524,288 (0.66) | 32,768 (0.041) | 4,096 (0.0052) | 4,096 (0.0052) | 1.00 | 1.00 |
| JPT vs rest of East Asia | 65,536 (0.083) | 32,768 (0.041) | 16,384 (0.021) | 524,288 (0.66) | never | 16,384 (0.021) | 65,536 (0.083) | never | 0.98 | 0.98 |
| FIN vs rest of Europe | 262,144 (0.33) | 16,384 (0.021) | 16,384 (0.021) | 262,144 (0.33) | 791,251 (1) | 262,144 (0.33) | 8,192 (0.01) | 4,096 (0.0052) | 0.96 | 0.97 |

Per-method slopes (S* vs N): JPT vs CHB: count-sketch -0.60, hashing 0.30, sparse JL -0.00, Misra–Gries 0.30, random SNP subset 0.30; JPT vs rest of East Asia: sparse JL -1.04, Misra–Gries 0.82, random SNP subset -0.68; FIN vs rest of Europe: count-sketch 1.46, hashing 0.40, sparse JL 0.10, Misra–Gries 1.00, random SNP subset 0.80.
Quantum register over the same N range: 17 → 18 → 19 → 20 index qubits (≈2× logical).

**Verdict: no memory advantage here either.** The signal is dense (top 1 % of SNPs hold only 12–15 % of ‖Δ‖²), but it is *compressible*: the best one-pass classical sketch needs 8,192–16,384 counters at N = 791,251 (S*/N = 0.01–0.021), and its memory barely grows with N (log–log slopes -0.68 to 0.40; 1 would mean ∝ N). Population differences are a few shared directions spread over many correlated SNPs, so random projections and hashing keep them. Dense is not the same as hard: the paper's worst cases need dense *and* incompressible signals.

- Same records for all methods: M = 100·N at every N (fixed after a first pass where the automatic M choice varied with N; first pass kept as `results/e8_route2_genotypes_autoM.json`, same conclusion). The paper's bound is on the product M·S.
- S* can fall as N grows (JPT vs rest of East Asia): more SNPs with the same per-SNP signal give a larger total margin, so a fixed-size sketch has more slack. More redundant features help the classical side too.
- Not a claim about accuracy: the quantum sketch at M → ∞ equals the exact centroid, which classical also computes with N floats; logistic regression on stored data is as good or better. The question is only memory in one pass.
- Limits: 1000 Genomes population labels are self-reported ancestry of the sampled groups, not clusters of the data (no circularity); test sets are small: ~31 positives per task; one test positive = 0.016 balanced accuracy; one split, 5 stream/hash seeds; Poissonised stream; the quantum column is a classical simulation; 6 chromosomes only (17–22); nested random SNP subsets, so N/8 ... N have the same per-SNP signal; close populations: logistic regression or PCA on stored data also work; the question is only one-pass memory; a finite experiment is evidence, not proof.
