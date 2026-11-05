> **Origin:** arena run 2026-11-05 14:04, cut short at the user's request. 7 independent plans written; 3 blind judges scored them on the arena rubric. This plan (a001) won all 3 verdicts (84, 84 and 83 out of 100; runner-up 81.5). No attack/defend rounds were run.
>
> **Execution:** carried out Thu 5 Nov 14:55–16:35 (see `plan_3_results.md` and the `qos_cells` repo). Deviations: gene selection by |μ₁−μ₀| of the squared-amplitude features (not Welch t); hardware calibration with the same oracle on training cells (not φ ≡ 0/±π/2); N=16 stretch run on Thursday; dashboard deferred to Friday.

# QSIP Hackathon – Plan 3

**Challenge:** ExtremeData4QML (data-centric QML), QSIP / ENCCS Stockholm
**Window:** Thursday 5 Nov, 14:00 → Friday 6 Nov, 5-minute talk in the afternoon
**One-line pitch:** *Single-cell RNA-seq is huge, sparse and full of rare cell types. A quantum computer can learn a rare-cell classifier by turning each streamed (cell, gene, label) record into one phase gate ("oracle sketching"). It keeps a log₂(genes)-qubit register in place of a genes-sized memory, and we measure exactly what class imbalance costs in samples.*

---

## 0. Decision summary (read this first)

| Item | Decision | Why (short) |
|---|---|---|
| Dataset | **PBMC3k single-cell RNA-seq** (starter dataset #9, `sc.datasets.pbmc3k()`), with rare cell types as the positive class | High-dimensional (32,738 genes), very sparse (97.4% zeros), low N (2,638 cells), and naturally imbalanced (dendritic cells 1.4%, megakaryocytes 0.6%). Three challenge properties at once, and it downloads in one line (5.6 MB). |
| Quantum technique | **Quantum oracle sketching** (Zhao, Zlokapa, Preskill, Huang et al., arXiv:2604.07639), as a streaming, imbalance-weighted, one-query classifier: the sketched diagonal unitary holds the class-difference vector as phases, and a Hadamard test reads out the score | The mechanism fits the data's properties (sparsity → one gate per nonzero, high dimension → log₂ N qubits, streaming → no stored dataset). It comes with a proven memory–sample trade-off. It is the person's thesis topic, so the Q&A defence is strong. |
| Our own contribution (novelty) | (1) **Imbalance-aware sketching**: class-weighted phase kicks, with the measured and derived sample cost ∝ 1/(π(1−π)). (2) An **honest resource ledger** that turns the paper's asymptotics into qubits, gates and seconds for a real atlas-scale problem. | The paper does not treat rare classes. Organisers asked for realism about qubits and speed. |
| Prototype | NumPy sketch simulator → Qiskit statevector circuit → shot simulator → IQM mock → **IQM real hardware, inference only**, with the trained phases fixed | Meets "simulator + toy on real IQM hardware" and the rule "no training on hardware". |
| Hardware toy | 3 qubits (2 data qubits = 4 genes, plus 1 Hadamard-test ancilla), about 7 CZ after transpilation; 8 test cells + 3 calibration circuits, 2,000 shots each | Shallow enough to survive noise. The calibration circuits detect and correct the score shrinkage that noise causes. |
| Extra products | One working model (the sketch classifier, `qos_cells` package) + a **TypeScript dashboard** (Vite + React + TS) that reads frozen JSON results | Matches the stated wants. The dashboard can never break the talk because it only reads files. |
| Fallbacks | Four degradation levels (L0–L3, Section 9). The talk exists at every level. | Defensive by design. |

---

## 1. Assumptions (stated, since nothing can be asked)

1. **Talk time:** "Friday afternoon" is taken as about 14:00. **Slides freeze Friday 12:30.** If the talk is earlier, move the Friday blocks up and cut the dashboard first (Section 8).
2. **Team:** the plan is written for **one person**. If there are teammates, the parallel lanes are marked **[B]** (dashboard/slides) and **[C]** (hardware liaison). Everything else is lane **[A]**.
3. **Machine:** a Windows laptop with Python 3.10–3.12 and internet. No GPU needed; everything here runs in seconds to minutes on a CPU.
4. **IQM access:** an `IQM_TOKEN` for Resonance is available, or can be had from the on-site IQM person. Emerald is blocked by the organisers (`qpu_pick.py` enforces this); we use whatever ready machine `qpu_pick` picks (Garnet-class, 20 qubits, is plenty).
5. **Fujitsu access** is a bonus. Nothing in the plan depends on it.
6. **Thesis code:** the paper's GitHub code is *optional reference*. The prototype below is self-contained (about 150 lines), so it does not rely on code that may not install on Windows tonight.
7. **"Training"** means accumulating the sketch phases from the data stream. That happens only in classical simulation. The hardware sees only fixed, already-computed phases. This is inference/validation, which the organisers allow.

---

## 2. Task 1 + 2: The dataset and what makes it hard

**Dataset:** PBMC3k (10x Genomics; 2,700 peripheral blood cells, 2,638 after standard QC, 32,738 genes). Cell-type labels come from `sc.datasets.pbmc3k_processed().obs["louvain"]`.

| Property | Number (checked in a dry run) | Why it hurts learning |
|---|---|---|
| High dimension, low N | 32,738 genes vs 2,638 cells | p ≫ n: overfitting, and memory grows with p |
| Sparsity | 97.4% of entries are zero | Most features are empty in most cells. The signal sits in a few genes |
| Class imbalance | CD4 T 1,144 · CD14 Mono 480 · B 342 · CD8 T 316 · NK 154 · FCGR3A Mono 150 · **Dendritic 37 (1.4%)** · **Megakaryocyte 15 (0.6%)** | Rare cell types are often the biologically important ones (rare immune states, circulating tumour cells), and they are what naive models miss |
| Streaming nature (in reality) | Atlases such as CZ CELLxGENE hold 10⁷–10⁸ cells; sequencers emit reads continuously | You cannot hold everything in memory. One-pass methods matter |

**Main framing for the talk:** *the hard part is not accuracy on 2,638 cells, which classical methods solve. The hard part is learning from a stream whose feature space is far bigger than the memory you can afford, while the class you care about is about 1% of the stream.* The quantum argument is aimed at exactly this pairing of memory and imbalance.

**Known caveat (say it before someone else does):** the Louvain labels come from clustering this same expression data, so labels and features are not independent. That is fine for a feasibility prototype, and it is mentioned on the limits slide.

---

## 3. Task 3: The quantum technique and the mechanistic argument

### 3.1 Mechanism in plain words

1. Each streamed record is one random entry of the data: (cell *i*, gene *g*, label *yᵢ*, value *xᵢg*).
2. It becomes **one diagonal phase gate** on a log₂N-qubit register:
   `V_t = exp( i · τ · wₜ · |g⟩⟨g| / M )`, where wₜ = +xᵢg²/π̂ for the rare class and −xᵢg²/(1−π̂) otherwise (importance weighting, the imbalance-aware part).
   The record is then **thrown away**.
3. All the gates are diagonal, so they commute. After M records, the register carries
   `U = diag( e^{iφ_g} )` with `φ_g ≈ c · (μ₁,g − μ₀,g)`, the class-mean difference of normalised squared expression. This is a phase oracle for a nearest-centroid linear classifier.
4. **Prediction for a new cell:** amplitude-encode the cell, |ψ⟩ = Σ_g a_g |g⟩, and run a Hadamard test on controlled-U. Then
   `⟨Z_anc⟩ = Σ_g a_g² · sin φ_g ≈ c · score(cell)`. The sign relative to a threshold gives the class.
5. In the full paper pipeline, classical shadows of these Hadamard tests give a *classical* model you can reuse. We implement the one-query version and say so.

Physical analogy for the talk: *each record gives the register a tiny "kick" in phase, like raindrops on a set of tuned drums. No drum stores the rain, but after a storm each drum is turned by an angle that reflects how much rain fell on it. The register remembers the sum, never the drops.*

### 3.2 Why this addresses *this* data (the mechanistic claims)

| Data property | Mechanism | Specific statement we can defend |
|---|---|---|
| High dimension N | Index register | **⌈log₂N⌉ qubits** for N features: 15 for all 32,738 genes; 30 for all ≈1.07×10⁹ gene–gene pair features (32,738² ≈ 2³⁰) |
| Sparsity | Gate per record | Zero entries create no gate, so cost scales with **nonzeros streamed**, not with N × cells |
| Streaming / memory | Oracle sketching | Paper's sample complexity for one oracle to error ε: **M = Θ(N/ε)**; for Q queries **M = Θ(N·Q²/ε)**. Classical lower bound (paper, Thm. 7 / NOPE task): **M·S ≥ Ω(N·Q_C)**, so a classical machine with the same samples needs memory S ≳ Q_C/Q² |
| Imbalance (our addition) | Importance-weighted kicks | Variance of the sketched phases grows as **1/(π(1−π))**: about 4 at π=0.5 and about 72 for dendritic cells (π=0.014). Dry run: the relative phase error at fixed M rose ×52 going from π=0.5 to π=0.005 (prediction ×50). **Rare classes cost samples, not qubits.** |
| Noise robustness of the sketch | Orthogonal supports | Gates for different genes act on orthogonal subspaces, so errors do not pile up across genes (paper's argument). Our check: infidelity ≈ N/M (N=16: M=10³→0.079, 10⁴→0.0098; N=256: M=10⁴→0.11, 10⁵→0.012) |

### 3.3 Claims register (to be defensive in Q&A)

| We **will** claim | We will **not** claim |
|---|---|
| The quantum memory is log₂N qubits for a sketch that would need N numbers classically *in this streaming model* | Any speed-up in wall-clock time |
| Sample cost scales as N/ε, with a measured 1/(π(1−π)) penalty for the rare class | That our centroid classifier provably inherits the paper's lower bound. That holds for the paper's task families; for our task it is an open question, and we say so |
| The hardware toy reproduces the simulator's scores after calibration (with the measured error) | That 4 genes on hardware is a useful classifier |
| For PBMC3k a classical model is better and cheap (logistic regression balanced accuracy ≈0.98 for B cells) | That we beat classical baselines (not required, and not true here) |

---

## 4. Task 4: The prototype (what to build, in order)

### 4.1 System map (how the parts depend on each other)

```
config.yaml (seed, genes N, positive class, M, tau, shots)  <- single source of truth
      |
 [D] load_data.py  -> normalised log1p matrix, labels, train/test split (stratified, seed)
      |                       \
 [S] stream.py  (random (cell,gene) records, class weights)   [B0] baselines.py (logreg, exact centroid)
      |
 [K] sketch.py  -> phases phi (N,)        <- golden test: Boolean oracle recovered
      |
 [C] circuits.py -> Qiskit circuits (state prep + controlled diag + Hadamard test)
      |            <- equivalence test: statevector <Z> == sum a^2 sin(phi)
 [X] backends: statevector | shot sim | IQM mock | IQM real | (Fujitsu)
      |
 [R] readout.py -> calibrated scores, metrics (balanced acc, AUPRC), error vs ideal
      |
 results/*.json (schema-versioned)  ---> slides figures (matplotlib)
                                     \--> TypeScript dashboard (reads same JSON)
```

**Second-order effects to manage (this is why the plan has gates):**
- **N (number of genes)** drives four things at once: qubits (log₂N), samples needed (∝N), hardware depth (a diagonal gate on n qubits costs about 2ⁿ gates), and accuracy. So use **N=4 on hardware, N=16 in shot simulation, N up to 1,024 in NumPy**. Never one N for all.
- **Small N creates all-zero cells.** With 2 genes, 96% of megakaryocyte test cells were all-zero (no quantum state exists). The data loader must **count and report** these and send them to an "abstain" bucket. Never silently drop them.
- **Noise shrinks ⟨Z⟩ toward 0.** That moves scores relative to the threshold. Calibration circuits are mandatory, not optional.
- **Phase wrap-around.** If any |φ_g| > π/2, sin φ stops being monotone and the classifier flips signs *silently*. Rescale c so that max|φ| ≤ π/2 and **assert** it.
- **Slides and dashboard must show the same numbers.** Both are generated from `results/*.json` and never typed by hand.

### 4.2 Repository layout

```
qos_cells\
  config.yaml
  qos_cells\__init__.py  load_data.py  stream.py  sketch.py  circuits.py  readout.py  resources.py  io.py
  scripts\01_baselines.py  02_scaling.py  03_imbalance.py  04_sim_circuits.py  05_hw_run.py  06_resources.py  07_figures.py  08_slides.py
  tests\test_golden.py
  results\   (JSON, git-tracked)
  figures\   presentation\plan3_template.html → plan3.html
  qpu_pick.py   (copied from the challenge repo)
  dashboard\   (Vite + React + TS, Friday)
```

### 4.3 Core code

See `qos_cells/sketch.py` (sketch_phases, rescale_safe, hadamard_score), `tests/test_golden.py` (Boolean oracle recovered, circuit == formula, shots, calibration) and `qos_cells/circuits.py` (Hadamard test with S†, controlled diagonal as one DiagonalGate on ancilla+register, ancilla = least-significant bit).

Calibration circuits: the frozen oracle applied to 4 *training* cells (2 rare, 2 common) + the zero sketch (ideal 0). Fit `Z_hw = α·Z_ideal + β` and invert. **Fail loud** if α < 0.3.

### 4.4 Experiments (each produces one JSON + one figure)

| # | Script | What it shows | Slide use |
|---|---|---|---|
| E1 | `01_baselines.py` | Logistic regression (class-balanced) and exact centroid on N∈{4,16,256,1024} genes, for B cells (13%), dendritic (1.4%), megakaryocytes (0.6%). Report **balanced accuracy and AUPRC** (never plain accuracy). Gene selection on the training split only | Honest baseline |
| E2 | `02_scaling.py` | Sketch phase error vs M for N∈{16,64,256,1024}: slope ≈ N/M; classifier accuracy vs M approaches the exact centroid | **Main mechanism plot** |
| E3 | `03_imbalance.py` | Phase error vs M for π∈{0.5,0.1,0.02,0.005} on a controlled synthetic, then on real B/DC/Mk. Overlay prediction 1/(π(1−π)) | **Our novelty plot** |
| E4 | `04_sim_circuits.py` | Same 8 test cells, N=4 and N=16: exact formula vs statevector vs 2,000-shot simulator vs IQM noisy fake backend vs remote mock | Feasibility chain |
| E5 | `05_hw_run.py` | N=4 (3 qubits) on IQM real hardware: raw ⟨Z⟩, calibrated ⟨Z⟩, error vs ideal, sign agreement. `--N 16` for the 5-qubit stretch | **Hardware slide** |
| E6 | `06_resources.py` | Resource ledger (Section 6) for N = 2¹⁵ and 2³⁰ | Real-world slide |

### 4.5 Hardware run procedure (inference only)

1. Phases φ for N=4 are computed **classically** (sketch of 100k records), frozen, and saved to `results/hw_phases.json` with a hash.
2. Build 8 test-cell circuits (4 positive, 4 negative, fixed indices) + 5 calibration circuits = 13 circuits.
3. Run the pipeline on a free mock first (`--mock`), then check the local shot simulator against the formula within 3σ. **Only then:**
4. Submit to the first ready machine in `config.hardware.preferred` (garnet, sirius); **save `job.job_id()` to disk immediately**.
5. Pull results, apply calibration, write `results/e5_hardware.json` (backend, timestamp, job id, raw counts).

---

## 5. Task 5: Real world

- **Where this data property occurs:** cell atlases (CZ CELLxGENE: tens of millions of cells), clinical flow and mass cytometry, and liquid biopsies, where circulating tumour cells are about 1 in 10⁶–10⁷ blood cells. All combine *huge feature spaces*, *one-pass or streaming acquisition*, and *rare but critical classes*.
- **Why it matters:** missing a rare population (a minimal residual disease clone, a rare immune state) is the expensive error. That is why the talk uses AUPRC, and why the cost of imbalance (×1/π in samples) is a design number, not a footnote.
- **Same structure elsewhere:** fraud (0.17%), AML transactions (0.1%), QEC decoding (0.14% logical errors). Imbalance-weighted sketching carries over unchanged. One sentence in the talk, no more.

---

## 6. Realism: the resource ledger

| Quantity | All genes (N = 2¹⁵ ≈ 32.7k) | Gene pairs (N = 2³⁰ ≈ 1.07×10⁹) |
|---|---|---|
| Index qubits | 15 | 30 |
| + Hadamard ancilla + work ancillas for multi-controlled phase (≈ n) | ≈ 30 logical | **≈ 60 logical** |
| Samples per query, M ≈ N·Q²/ε (ε=0.1, Q=1) | ≈ 3.3×10⁵ | ≈ 1.1×10¹⁰ |
| × imbalance factor 1/(π(1−π)), dendritic (π=0.014) | ≈ 2.4×10⁷ | ≈ 7.8×10¹¹ |
| Gates per sketch (≈ M multi-controlled phases, ≈ 2n−1 each) | ≈ 7×10⁸ | ≈ 5×10¹³ |
| Time at 1 MHz logical gates | ≈ 700 s | ≈ 1.5 years |
| Classical memory for the same sketch | 131 kB | ≈ 4.3 GB |

**The honest conclusion:** this needs **fault-tolerant** logical qubits with fast logical gates, and the advantage is in **memory/size, not time**. At 15 genes' worth of qubits classical wins easily. The argument becomes interesting only at feature spaces of about 10⁹ and more, such as gene-pair interactions and multi-omics. The hardware toy shows the circuit primitive, not usefulness.

---

## 7. Deliverables

### 7.1 Slides (7 talk slides + 3 backup, `presentation/plan3.html`)

1. **Hook** + five-step pipeline. 2. **Data + challenge.** 3. **Mechanism** (drum analogy, equation, Hadamard readout, our ±1/π̂ addition). 4. **Why it fits** (claims table + E2 plot). 5. **Our twist** (E3 plot, 1/(π(1−π))). 6. **It runs** (chain + IQM Garnet bars, 11 CZ vs 53 CZ). 7. **Real world + ledger + claims register.** Backup: E1 baselines, Q&A register, hardware/calibration details.

### 7.2 Q&A preparation

| Question | Answer |
|---|---|
| "Classical does this with 14k floats." | Yes, and we show it. The claim is about the regime N ≫ memory (gene pairs, 10⁹ features), where the paper proves a memory–sample trade-off. |
| "Does your centroid task inherit the lower bound?" | Not proven. The paper's bound is for its task families (NOPE, forrelation-type). We say it is open. |
| "Loading data kills quantum advantages." | Sketching *is* the loading. Each record is one gate and is never stored, so there is no QRAM. The cost is samples ∝ N/ε, which we show. |
| "Hardware noise?" | Known-answer circuits of the same family measure the shrink factor α. We report raw and calibrated values; 11 CZ works raw, 53 CZ needs the calibration. |
| "Why not train on hardware?" | Against the rules. Training (sketch accumulation) is classical simulation; hardware runs inference with fixed phases. |
| "Is this just a linear model?" | Yes, on purpose. A one-query sketch gives a linear functional. Nonlinear models need more queries Q, and the cost grows as Q², which is the paper's optimal scaling. |

### 7.3 Working model

`qos_cells` package + `scripts/` + `tests/`: `pytest` passes (8 tests), and `python scripts/04_sim_circuits.py` reproduces the scores from a clean clone in under a minute.

### 7.4 TypeScript dashboard (Friday, time-boxed)

- **Stack:** Vite + React + TypeScript + Recharts. No backend: it reads `public/results/*.json`, copied from `results/`.
- **Panels (priority order):** 1. Resource estimator (sliders N, ε, Q, π → qubits, samples, gates, time, classical memory). 2. E2 curves. 3. E3 curves. 4. E5 bars.
- **Defensive:** every JSON has `"schema_version"`; a mismatch shows a red banner naming the file. Works offline from `dist/`.
- **Use in the talk:** a screenshot at most; never a dependency.

---

## 8. Friday schedule

| Time | Work | Gate |
|---|---|---|
| 08:30–09:30 | Optional: re-run `05_hw_run.py` (day-to-day drift of α) or run the 13 circuits on Sirius | — |
| 09:30–11:30 | Dashboard panels 1→4 | **G6 (11:30):** panel 1 + one chart built to `dist/`; else screenshot what works |
| 11:30–12:30 | Final slide polish; claims register in speaker notes | **Freeze 12:30.** PDF export (Ctrl+P, landscape); two copies |
| 12:30–13:15 | Rehearse three times aloud with a timer (target 4:30) | ≤ 5:00 on the third run |
| 13:15–talk | Lunch, buffer, projector adapter | — |

---

## 9. Degradation ladder (the talk exists at every level)

| Level | Condition | What we present |
|---|---|---|
| **L0** | Everything works | Full story: E1–E6, hardware, dashboard |
| **L1** | Hardware JSON lost | Same story from the IQMFakeApollo noisy simulation, labelled clearly, with the job ids |
| **L2** | Sketch scaling on real data unstable | Exact-phase classifier on real data + synthetic scaling + theory |
| **L3** | Code broadly broken | Theoretical talk: mechanism, resource ledger, imbalance derivation |

---

## 10. Why this plan (argument against the alternatives)

| Alternative | Why not chosen |
|---|---|
| Google surface-code data | Zenodo format + decoder-aware model in 24 h is risky; classical (AlphaQubit) context is strong |
| `qspin` + symmetry-equivariant circuits | Crowded idea; does not use the thesis expertise |
| Credit-card fraud | PCA-anonymised features remove the structure to argue about |
| IBM AML graph | 5M edges and graph-to-circuit design do not fit the time box |
| Generic VQC/kernel | Explicitly discouraged by the organisers |

---

## 11. Checklist before walking on stage

- [x] `pytest` green; all figures regenerated from `results/*.json` after the last code change
- [x] Every number on the slides appears in a JSON file (no hand-typed numbers)
- [x] Hardware slide states: machine name, date, shots, raw vs calibrated, job id, "inference only, fixed parameters"
- [x] Metrics are balanced accuracy / AUPRC only
- [x] Realism slide states: qubits, samples, gates, "fault-tolerant needed", "memory not speed"
- [x] Claims register on the slides and in `plan_3_results.md`
- [ ] Talk ≤ 5:00 timed; PDF on two devices; adapter checked
