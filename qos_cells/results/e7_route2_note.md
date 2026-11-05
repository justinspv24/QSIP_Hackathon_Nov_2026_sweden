## Route 2 (E7): classical memory stress test on gene-pair features

**Question.** The quantum sketch keeps ⌈log₂N⌉ qubits for N features. Can a classical *streaming* method with memory S ≪ N already solve the rare-cell task? Script `scripts/09_route2_classical_memory.py`, results `results/e7_route2.json`, figure `figures/fig_e7_memory.png`, backup slide 11 of `presentation/plan3.html`.

**Verdict at K = 512 (N = 131,328 pair features, 18 index / ≈36 logical qubits):** memory advantage on PBMC3k = **no**. Worst class needs S*/N = 1e-3 with the best one-pass sketch; the structureless control needs S*/N ≥ 1.00 (best one-pass sketch: heavy hitters (MG)); the others never reach the band below S = N.

**Verdict at K = 2048 (N = 2,098,176 pair features, 22 index / ≈44 logical qubits):** memory advantage on PBMC3k = **no**. Worst class needs S*/N = 1e-4 with the best one-pass sketch; the structureless control needs S*/N ≥ 0.12 (best one-pass sketch: heavy hitters (MG)); the others never reach the band below S = N.

S* = smallest S whose mean balanced accuracy (5 stream/hash seeds, same records, same ±1/π̂ weights, same split) stays within tol of the exact centroid for all larger S; tol = max(0.01, one test positive). The strict-0.01 S*, the S* against the S = N stream estimate at the same M, and the AUPRC version are all in the JSON. Numbers in brackets are S*/N.


**K = 512, N = 131,328** (stream M = M_run·N records per task; quantum: 18 index qubits, classical simulation only)

| task | exact centroid (bal. acc / AUPRC) | tol | S = N stream | count-sketch | feature hashing | sparse JL | heavy hitters (MG) | reservoir | oracle top-S | hashing + logreg (not one-pass) |
|---|---|---|---|---|---|---|---|---|---|---|
| B cells | 0.96 / 0.96 | 0.01 | 0.96 at M = 100·N | 2^8 (2e-3) | 2^7 (1e-3) | 2^8 (2e-3) | 2^9 (4e-3) | never (—) | 8 (6e-5) | 2^7 (1e-3) |
| Dendritic cells | 0.92 / 0.48 | 0.05 | 0.91 at M = 100·N | 2^6 (5e-4) | 32 (2e-4) | 2^6 (5e-4) | 2^8 (2e-3) | 2^14 (0.12) | 8 (6e-5) | 16 (1e-4) |
| Megakaryocytes | 0.90 / 0.69 | 0.10 | 0.87 at M = 2700·N | 32 (2e-4) | 32 (2e-4) | 8 (6e-5) | 16 (1e-4) | 131,328 (1.00) | 32 (2e-4) | 32 (2e-4) |
| synthetic | 1.00 / 1.00 | 0.02 | 1.00 at M = 8100·N | never (—) | never (—) | never (—) | 2^17 (1.00) | never (—) | 2^16 (0.50) | — |

Logistic regression on all N features (class-balanced, sanity check): B 0.99/0.99, Dendritic 0.95/0.98, Megakaryocytes 0.80/0.79.
Simulated quantum readout p·sin(c·φ) at M_run: B 0.96, Dendritic 0.91, Megakaryocytes 0.87 (same information as the S = N stream column).

**K = 2048, N = 2,098,176** (stream M = M_run·N records per task; quantum: 22 index qubits, classical simulation only)

| task | exact centroid (bal. acc / AUPRC) | tol | S = N stream | count-sketch | feature hashing | sparse JL | heavy hitters (MG) | reservoir | oracle top-S | hashing + logreg (not one-pass) |
|---|---|---|---|---|---|---|---|---|---|---|
| B cells | 0.95 / 0.94 | 0.01 | 0.95 at M = 100·N | 2^9 (2e-4) | 2^8 (1e-4) | 2^8 (1e-4) | 2^9 (2e-4) | 2^19 (0.25) | 8 (4e-6) | 2^7 (6e-5) |
| Dendritic cells | 0.92 / 0.40 | 0.05 | 0.92 at M = 100·N | 2^8 (1e-4) | 2^7 (6e-5) | 32 (2e-5) | 2^6 (3e-5) | 2^17 (6e-2) | 16 (8e-6) | 16 (8e-6) |
| Megakaryocytes | 0.80 / 0.69 | 0.10 | 0.83 at M = 100·N | 16 (8e-6) | 8 (4e-6) | 8 (4e-6) | 2^7 (6e-5) | 2^12 (2e-3) | 32 (2e-5) | 8 (4e-6) |
| synthetic | 0.99 / 1.00 | 0.06 | 0.99 at M = 8100·N | never (—) | 2^21 (1.00) | 2^20 (0.50) | 2^18 (0.12) | never (—) | 2^16 (3e-2) | — |

Logistic regression on all N features (class-balanced, sanity check): B 0.99/0.99, Dendritic 0.95/0.96, Megakaryocytes 0.80/0.75.
Simulated quantum readout p·sin(c·φ) at M_run: B 0.95, Dendritic 0.92, Megakaryocytes 0.81 (same information as the S = N stream column).

**What it means for the advantage claim.**
- On PBMC3k the class signal lives in a few marker-gene pairs: the oracle top-8 features already reproduce the exact centroid for B cells at both N, and every linear sketch (count-sketch, hashing, JL) matches it with a few hundred counters — S* = 128–256 at N = 1.3·10⁵ and 256–512 at N = 2.1·10⁶, i.e. S*/N ≈ 10⁻³ → 10⁻⁴. The memory needed does not grow with N (16× more features, ≤ 2× more counters); 2–4 kB of classical counters do what the 18–22-qubit register (≈36–44 logical) does, on the same stream.
- The control shows what the paper's claim needs: with the same signal spread evenly over all N features, the linear sketches reach the band only at S ≈ N/2–N and heavy hitters at S ≈ N/8–N (N/4–N under the strict 0.01 rule), while the exact centroid still classifies (0.99). That is the regime of the paper's lower bounds (structureless worst-case families). Real scRNA-seq is not in that regime.
- The big feature space is not where the signal is: the exact centroid gets worse with gene pairs than with 16 genes (E1: B 0.97 → 0.95, DC 0.98 → 0.92, Mk 1.00 → 0.80 at N = 2.1·10⁶); logistic regression recovers B cells (0.99) but not megakaryocytes (0.80). The sketch also needs M ≈ 100·N ≈ 2·10⁸ records before its own classification converges at this N.
- So Route 2 confirms the earlier framing: claim level stays *evidence for the mechanism*, not a demonstrated advantage. An advantage on real data would need a task whose decision direction is dense over an N ≫ memory feature space — rare-cell markers are the opposite of that.
- Limits: Louvain labels are circular; DC/Mk test sets are 11/5 cells (one cell = 0.045/0.10 balanced accuracy, hence the per-class tolerance; the strict-0.01 S* for these classes is in the JSON and is noise-dominated); one split (config seed) with 5 stream/hash seeds; the stream is Poissonised (independent Poisson counts per (cell, feature) entry) instead of record by record; the synthetic control at N = 2.1·10⁶ has only 160 training / 80 test cells (memory); the quantum column is a classical simulation at this N; a finite experiment is evidence, not proof.
