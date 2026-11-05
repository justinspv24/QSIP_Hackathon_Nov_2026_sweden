# QSIP Hackathon – Plan 3 results and talk notes

**Status:** executed Thu 5 Nov 2026, 14:55–16:30. All gates G0–G5 passed on Thursday (the plan expected G5 on Friday morning).
**Code:** `qos_cells` repo (zip in chat; local copy in `C:\Users\justi\Projects\QSIP\qos_cells`). Deck: `presentation/plan3.html` (10 slides: 7 talk + 3 backup).
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
