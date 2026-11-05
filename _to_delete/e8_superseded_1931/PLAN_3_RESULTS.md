# QSIP Hackathon – Plan 3 results and talk notes

**Status:** executed Thu 5 Nov 2026, 14:55–16:30. All gates G0–G5 passed on Thursday (the plan expected G5 on Friday morning).
**Code:** `qos_cells` repo (zip in chat; local copy in `C:\Users\justi\Projects\QSIP\qos_cells`). Deck: `presentation/plan3.html` (11 slides: 7 talk + 4 backup; slide 11 = Route 2).
**Rule kept:** every number on the slides is read from `results/*.json` by `scripts/08_slides.py`; nothing is typed by hand.

## Gate log

| Gate | Plan deadline | Done | Outcome |
|---|---|---|---|
| G0 token + machine | Thu 14:35 | Thu 15:05 | Resonance API reachable; garnet (20 q) and sirius (16 q) online, queue 0; emerald blocked by the organisers |
| G1 data + E1 | Thu 15:45 | Thu 15:12 | PBMC3k (cellxgene h5ad) loads in one line; E1 JSON written |
| G2 golden test + E2 + E3 | Thu 17:15 | Thu 15:16 | 8/8 tests pass; E2 slope matches N/M; E3 within ×1.4 of the law |
| G3 circuits + mock | Thu 18:30 | Thu 15:20 | statevector = formula to 1e-15; shots within 0.034; mock pipeline OK |
| G5 calibrated hardware scores | **Fri 09:30** | **Thu 16:17** | Garnet, 3 qubits, 11 CZ: α = 0.78, 8/8 correct. Stretch 5 qubits / 53 CZ also done |
| G4 slides rough | Thu 21:30 | Thu 16:30 | full deck built from JSON, render-checked in headless Chrome |

## Headline numbers (from results/*.json)

| Experiment | Result |
|---|---|
| Data | 2,638 cells × 13,714 expressed genes (32,738 measured), 93.8 % zeros; DC 37 (1.4 %), Mk 15 (0.6 %) |
| E1 exact centroid, N=16 (bal. acc / AUPRC) | B 0.97 / 0.98 · DC 0.98 / 0.75 · Mk 1.00 / 1.00. Logreg on all 13,714 genes: B 0.97 / 1.00, DC 0.86 / 0.96, Mk 0.80 / 0.77 |
| E2 N/M law | err² ≈ C·N/M, C ≈ 90 for N ≥ 256; closed-form prediction with no fit tracks the 5-seed measurement |
| E2 classifier, N=1024 (10 qubits) | bal. acc 0.68 (M=10³) → 0.83 (10⁴) → 0.94 (10⁵) → 0.96 (3·10⁵) = exact centroid 0.96 |
| E2 sparsity | training density 35 % at N=1024; nonzero-only streaming reaches a given error with ~2× fewer records |
| E3 rare-class cost | measured ×3.1 / ×13.5 / ×71 for π = 0.1 / 0.02 / 0.005 (law ×2.8 / ×12.8 / ×50). Real: B ×9 (4.1·10⁵ records for 10 % error), DC ×72 (2.4·10⁶), Mk ×186 (1.5·10⁶, strong markers) |
| E4 chain | formula = statevector (1e-15); 2,000 shots ±0.03; IQMFakeApollo noise shrinks scores to ≈0.7 (N=4) and ≈0.4 (N=16) |
| E5 Garnet N=4 | 3 qubits, 11 CZ, depth 24, 2,000 shots: α = 0.776, β = 0.040; mean error 0.127 raw → 0.046 calibrated; **8/8 cells correct raw and calibrated**; job 01a1ac11-9106-73d2-9760-b430c7575476 (2026-11-05 14:16 UTC) |
| E5 Garnet N=16 (stretch) | 5 qubits, 53 CZ, depth 108: α = 0.431; mean error 0.258 raw → 0.094 calibrated; **raw 5/8 → calibrated 8/8**; job 01a1ac12-f993-75ba-9166-23625a6bf9f9 |
| E6 ledger | all genes: 15 index / ≈30 logical qubits, 2.4·10⁷ records at π = 0.014, ≈700 s at 1 MHz, classical 131 kB. Gene pairs: 30 / ≈60 qubits, 7.8·10¹¹ records, ≈1.5 years per sketch, classical 4.3 GB |

## Deviations from the plan (and why)

1. **Gene selection** is by |μ₁−μ₀| of the squared-amplitude features on the training split, not by a Welch t on log-expression. The t-statistic picked genes megakaryocytes never express (all 10 training positives became all-zero cells); the |Δ| rule is aligned with the model's own weights, gives 0 % abstain, and picks the biology (PPBP/PF4 for Mk, CD74/HLA for B cells).
2. **Hardware calibration** uses the frozen oracle on 4 training cells + the zero sketch (same gate structure as the test circuits). The plan's φ ≡ 0 / ±π/2 circuits would transpile to trivial circuits; the first attempt with random-phase circuits over-corrected (kept as `results/e5_hardware_run1_randomcal.json`, backup slide).
3. **Remote mocks are random-bit generators** on Resonance, so the noisy simulator is the local IQMFakeApollo model; the mocks only test the job pipeline.
4. **N=16 stretch** was run on Thursday because Garnet had an open pay-as-you-go window and zero queue.
5. **Dashboard** (Vite + React + TypeScript + Recharts) was built on Thursday after all: `qos_cells/dashboard/` with the four planned panels (resource estimator with sliders, E2, E3, E5); `npm run build` bundles `results/*.json` in and `dashboard/dist/index.html` opens offline. A schema mismatch shows a red banner naming the file.

## 5-minute talk — speaker notes (≈40 s per slide)

1. **Hook.** 1 in 70 blood cells is a dendritic cell; 1 in a million a tumour cell. Atlases hold 10⁸ cells; you cannot store the stream. Can a small register learn the rare class from the stream? Five steps, one dataset.
2. **Data.** PBMC3k: 32k genes measured, 94 % zeros, 2,638 cells, two classes under 2 %. The hard part is not accuracy here, classical gets that; it is memory ≈ features, a one-pass stream, and a 1 % signal.
3. **Mechanism.** Each record is one phase gate on log₂N qubits; the gates commute, the register keeps the sum; a Hadamard test reads Σ a² sin φ, a linear score. Drum analogy. Our addition: ±1/π̂ weights make the sketch imbalance-aware.
4. **Why it fits.** Table of claims, each with a measurement: log₂N qubits, N/ε samples, zeros are free. Plot: error vs M for four N, dashed lines predicted with no fit. Classifier at 10 qubits reaches the centroid by 3·10⁵ records.
5. **Our twist.** Rare classes cost samples, not qubits: two-line variance derivation gives 1/(π(1−π)); measured ×71 vs law ×50 at π = 0.005; dendritic pays ×72. Rare and subtle is the expensive combination.
6. **It runs.** Formula = statevector = shots; then IQM Garnet: 3 qubits, 11 CZ, 8/8 cells right, α = 0.78. At 53 CZ raw signs break (5/8) and calibration rescues them (8/8). Inference only, phases frozen and hashed, job ids on the slide.
7. **Real world + ledger.** Atlases, cytometry, liquid biopsies; fraud/AML/QEC share the shape. Ledger: 30 logical qubits and 700 s for all genes (classical wins), 60 qubits and ~1.5 years per sketch for gene pairs vs 4 GB classical. Memory not time; needs fault tolerance; we do not claim the lower bound for our task.

## Q&A register

See backup slide 9 (seven questions with answers). Key lines: "sketching *is* the loading, no QRAM"; "the bound is proven for the paper's tasks, open for ours"; "linear on purpose: Q queries cost Q²"; "labels are Louvain clusters of the same data".

## Friday checklist

- [x] Dashboard built (`qos_cells/dashboard/dist/index.html`); re-run `npm run build` only if results change
- [ ] Rehearse 3× aloud with a timer (target 4:30); PDF export of `plan3.html` (Ctrl+P, landscape) on two devices
- [ ] Optional: Sirius run of the same 13 circuits (different topology, MOVE gates) as a second-device data point
- [ ] Optional: re-run `05_hw_run.py` Friday morning to show day-to-day drift of α

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
