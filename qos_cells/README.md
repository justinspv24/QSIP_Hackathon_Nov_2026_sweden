# qos_cells — quantum oracle sketching for rare-cell classification

QSIP Hackathon 2026 (ENCCS Stockholm), ExtremeData4QML challenge, Plan 3.
One-line pitch: *single-cell RNA-seq is huge, sparse and full of rare cell types; a quantum register of
log₂(genes) qubits can learn a rare-cell classifier from the record stream (one phase gate per record), and
we measure exactly what class imbalance costs in samples.*

Paper: Zhao, Zlokapa, Preskill, Huang (2026), "Exponential quantum advantage in processing massive classical data",
arXiv:2604.07639.

## Run everything from a clean clone

```bash
pip install "iqm-client[qiskit]" numpy scipy scikit-learn matplotlib pytest pyyaml anndata h5py qiskit-aer
python -m pytest -q tests                 # 8 golden tests (Boolean oracle recovered, circuit == formula, ...)
python scripts/01_baselines.py            # E1  classical baselines (PBMC3k downloads itself, ~25 MB)
python scripts/02_scaling.py              # E2  the N/M law + classifier vs M + sparsity
python scripts/03_imbalance.py            # E3  the 1/(pi(1-pi)) rare-class cost
python scripts/04_sim_circuits.py         # E4  formula == statevector == shots == noisy fake backend; freezes phases
export IQM_TOKEN=...                      # (PowerShell: $env:IQM_TOKEN="...")
python scripts/05_hw_run.py               # E5  IQM Garnet, 3 qubits, inference only  (--N 16 for the 5-qubit stretch)
python scripts/06_resources.py            # E6  resource ledger
python scripts/07_figures.py              # figures/*.png from results/*.json
python scripts/08_slides.py               # presentation/plan3.html from results/*.json (no hand-typed numbers)
```

Everything is driven by `config.yaml`; every `results/*.json` records the config it was produced with and a
`schema_version`. The slides and the dashboard only read those files.

## What the code does

| module | role |
|---|---|
| `qos_cells/load_data.py` | PBMC3k (cellxgene h5ad: louvain labels + log1p-normalised raw), stratified split, **train-only** gene selection by \|μ₁−μ₀\| of the squared-amplitude features, amplitude features p = x²/‖x‖², abstain bucket for all-zero cells |
| `qos_cells/stream.py` | the record stream (cell, gene, label, value) → (gene index, importance-weighted kick ±p/π̂); closed-form mean and variance of the sketch (parameter-free predictions); synthetic pools with a controlled class fraction |
| `qos_cells/sketch.py` | the oracle sketch: φ_g = (τ/M) Σ_{t: g_t=g} w_t; safe rescaling to max\|φ\| = π/2 (no silent sign flips); Hadamard-test score Σ a² sin φ |
| `qos_cells/circuits.py` | Qiskit: state preparation + controlled diagonal + Hadamard test with S†; statevector check; shot simulator; IQM transpilation |
| `qos_cells/readout.py` | balanced accuracy / AUPRC only, threshold on training scores, calibration fit z_hw = α z_ideal + β (fail loud if α < 0.3) |
| `qos_cells/resources.py` | the ledger: qubits, samples, gates, seconds for N = 2¹⁵ and 2³⁰ |

## Results (5 Nov 2026)

* **Data:** 2,638 cells × 13,714 expressed genes (32,738 measured), 93.8 % zeros; dendritic cells 37 (1.4 %), megakaryocytes 15 (0.6 %).
* **E1:** exact centroid (= M→∞ sketch) balanced accuracy / AUPRC at N=16: B cells 0.97/0.98, dendritic 0.98/0.75, megakaryocytes 1.00/1.00. Rare classes prefer *few* genes (p ≫ n).
* **E2:** sketch error follows the closed-form prediction with no fitted parameter; err² ≈ C·N/M with C ≈ 90 for N ≥ 256. Classifier at N=1024 (10 qubits) reaches the exact centroid (bal. acc 0.96) by M ≈ 3×10⁵ records.
* **E3:** sample cost ∝ 1/(π(1−π)): measured ×3.1 / ×13.5 / ×71 for π = 0.1 / 0.02 / 0.005 vs the law's ×2.8 / ×12.8 / ×50; closed-form prediction within 10 %. Dendritic cells pay ×72.
* **E5 (real hardware, IQM Garnet, 2026-11-05):** N=4, 3 qubits, 11 CZ: α = 0.78, 8/8 cells correct raw and calibrated (job `01a1ac11-9106-73d2-9760-b430c7575476`). N=16, 5 qubits, 53 CZ: α = 0.43, raw 5/8 → calibrated 8/8 (job `01a1ac12-f993-75ba-9166-23625a6bf9f9`). Inference only; phases frozen with a hash.
* **E6:** all genes: 15 index / ≈30 logical qubits, 2.4×10⁷ records (π = 0.014), ≈ 700 s at 1 MHz logical gates, classical 131 kB. Gene pairs: 30 / ≈60 qubits, 7.8×10¹¹ records, ≈ 1.5 years per sketch, classical 4.3 GB. Memory, not time; needs fault tolerance; classical wins at this scale.

## Honest notes

* The remote free mocks on Resonance (`garnet:mock`, `emerald:mock`) return uniformly random bits; they validate the job pipeline only. The local `IQMFakeApollo` noise model is what we use as the noisy simulator.
* Hardware run 1 used random-phase calibration circuits on one cell; their ideal scores bunched near 0 and the fit over-corrected. Run 2 calibrates with the same oracle on 4 training cells + the zero sketch. Both JSONs are kept.
* Louvain labels come from clustering the same expression data (labels and features are not independent).
* The paper's classical lower bound is for its task families; whether our centroid task inherits it is open.
* In a real stream τ is fixed in advance (from the bound p ≤ 1 or a pilot batch); here the sketch is simulated, so the rescaling is applied afterwards and reported as c.
