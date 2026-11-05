> **Origin:** arena run 2026-11-05 14:04, cut short at the user's request. 7 independent plans written; 3 blind judges scored them on the arena rubric. This plan (a001) won all 3 verdicts (84, 84 and 83 out of 100; runner-up 81.5). No attack/defend rounds were run.

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

### 4.2 Repository layout (create at 14:15)

```
C:\qsip\qos_cells\
  config.yaml
  qos_cells\__init__.py  load_data.py  stream.py  sketch.py  circuits.py  readout.py  resources.py
  scripts\01_baselines.py  02_scaling.py  03_imbalance.py  04_sim_circuits.py  05_hw_run.py  06_figures.py
  tests\test_golden.py
  results\   (JSON, git-tracked)
  qpu_pick.py   (copied from the challenge repo)
  dashboard\   (Vite + React + TS)
```

### 4.3 Core code (self-contained, tested in a dry run)

```python
# qos_cells/sketch.py
import numpy as np

def sketch_phases(gene_idx, weights, N, M, tau):
    """Oracle sketching for a diagonal unitary.
    Each record (g, w) is the gate exp(i*tau*w/M |g><g|). The gates commute,
    so the product is diag(exp(i*phi)). Returns phi."""
    gene_idx = np.asarray(gene_idx); weights = np.asarray(weights, float)
    if gene_idx.min() < 0 or gene_idx.max() >= N:
        raise ValueError("gene index out of range")
    phi = np.zeros(N)
    np.add.at(phi, gene_idx, tau * weights / M)
    if not np.all(np.isfinite(phi)):
        raise FloatingPointError("non-finite phase")
    return phi

def rescale_safe(phi, limit=np.pi / 2):
    """Keep sin() monotone. Returns (phi_scaled, c). Fail loud on an all-zero sketch."""
    m = np.abs(phi).max()
    if m == 0:
        raise ValueError("sketch is all zero: stream empty or labels missing")
    c = limit / m
    return phi * c, c

def hadamard_score(a, phi):
    """Ideal <Z_anc> of the Hadamard test (with S-dagger), = sum a_g^2 sin(phi_g)."""
    a = np.asarray(a, float); n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("all-zero cell on selected genes -> abstain")
    return float(np.sum((a / n) ** 2 * np.sin(phi)))
```

```python
# tests/test_golden.py  -- must pass before anything else is trusted
import numpy as np
from qos_cells.sketch import sketch_phases
def test_boolean_oracle_recovered():
    rng = np.random.default_rng(0); N = 16; M = 200_000
    f = rng.integers(0, 2, N); x = rng.integers(0, N, M)
    phi = sketch_phases(x, f[x], N, M, tau=np.pi * N)
    assert np.allclose(np.sign(np.cos(phi)), (-1.0) ** f)   # paper's phase oracle (-1)^f
```

```python
# qos_cells/circuits.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate

def hadamard_test_circuit(a, phi):
    """q0 = ancilla, q1..qn = gene register. Measures <Z_q0> = sum a^2 sin(phi)."""
    N = len(a); n = int(np.log2(N)); assert 2 ** n == N, "N must be a power of 2"
    a = np.asarray(a, float) / np.linalg.norm(a)
    qc = QuantumCircuit(n + 1, 1)
    qc.prepare_state(list(a), list(range(1, n + 1)))
    qc.h(0)
    d = np.ones(2 ** (n + 1), complex)          # controlled-diag as one diagonal on anc+data
    for k in range(N):
        d[2 * k + 1] = np.exp(1j * phi[k])      # ancilla is the least-significant bit
    qc.append(DiagonalGate(list(d)), list(range(n + 1)))
    qc.sdg(0); qc.h(0); qc.measure(0, 0)
    return qc
```

Dry-run facts: statevector ⟨Z⟩ equals Σa²sin φ to 4 decimals. After transpiling to IQM's native `r`+`cz`: **2 qubits → 2 CZ, depth 9; 3 qubits → 7 CZ, depth 20.**

Calibration circuits (same shape, any fixed cell): φ ≡ 0 → ideal 0; φ ≡ +π/2 → ideal +1; φ ≡ −π/2 → ideal −1. Fit `Z_hw = α·Z_ideal + β` and invert. **Fail loud** if α < 0.3, since the device is too noisy for a sign readout; then switch qubits or the machine.

### 4.4 Experiments (each produces one JSON + one figure)

| # | Script | What it shows | Slide use |
|---|---|---|---|
| E1 | `01_baselines.py` | Logistic regression (class-balanced) and exact centroid on N∈{4,16,256,1024} genes, for B cells (13%), dendritic (1.4%), megakaryocytes (0.6%). Report **balanced accuracy and AUPRC** (never plain accuracy). Gene selection on the training split only | Honest baseline |
| E2 | `02_scaling.py` | Sketch phase error vs M for N∈{16,64,256,1024}: slope ≈ N/M; classifier accuracy vs M approaches the exact centroid | **Main mechanism plot** |
| E3 | `03_imbalance.py` | Phase error vs M for π∈{0.5,0.1,0.02,0.005} on a controlled synthetic, then on real B/DC/Mk. Overlay prediction 1/(π(1−π)) | **Our novelty plot** |
| E4 | `04_sim_circuits.py` | Same 8 test cells, N=4 and N=16: exact formula vs statevector vs 2,000-shot simulator vs IQM mock | Feasibility chain |
| E5 | `05_hw_run.py` | N=4 (3 qubits) on IQM real hardware: raw ⟨Z⟩, calibrated ⟨Z⟩, error vs ideal, sign agreement | **Hardware slide** |
| E6 | `resources.py` | Resource ledger (Section 6) for N = 2¹⁵ and 2³⁰ | Real-world slide |

Dry-run reference values for sanity checks (yours will differ a little, but large deviations mean a bug):
- B cells, N=16 genes picked by |w| on train: exact centroid balanced accuracy ≈0.93, AUPRC ≈0.99.
- Dendritic, N=16: bal. acc ≈0.83, AUPRC ≈0.58 (base rate 0.014).
- Megakaryocytes, N=16: bal. acc ≈0.90, AUPRC ≈0.97 (base rate 0.006).
- Sketch vs exact at N=1,024 (variance-selected genes), B cells: M=10³ is at chance (0.46), 10⁴ gives 0.97, 10⁵ gives 0.98, the same as exact. **This is the N/M law visible in classification.**

### 4.5 Hardware run procedure (inference only)

1. Phases φ for N=4 are computed **classically**, frozen, and saved to `results/hw_phases.json` with a hash.
2. Build 8 test-cell circuits (4 positive, 4 negative, fixed indices in config) + 3 calibration = 11 circuits.
3. `backend, tc = pick_and_transpile(circuits, mock=True)`, then run and check that results match the shot simulator within 3σ. **Only then:**
4. `python qpu_pick.py --qubits 3` to see queues; then `pick_and_transpile(circuits)`; `job = backend.run(tc, shots=2000)`. **Save `job.job_id()` to disk immediately**, so a crash does not lose the job.
5. Pull results, apply calibration, write `results/hw_run.json` (with backend name, timestamp, job id, raw counts).
6. Stretch (only if step 5 is done by Friday 10:00): N=16 (5 qubits, deeper) or the same circuits on Fujitsu via QASM export.

Shot budget: 11 × 2,000 = 22,000 shots. Statistical error on ⟨Z⟩ ≈ 1/√2000 ≈ 0.022, enough to resolve test scores of magnitude ≈0.1–0.5.

---

## 5. Task 5: Real world

- **Where this data property occurs:** cell atlases (CZ CELLxGENE: tens of millions of cells), clinical flow and mass cytometry, and liquid biopsies, where circulating tumour cells are about 1 in 10⁶–10⁷ blood cells. All combine *huge feature spaces*, *one-pass or streaming acquisition*, and *rare but critical classes*.
- **Why it matters:** missing a rare population (a minimal residual disease clone, a rare immune state) is the expensive error. That is why the talk uses AUPRC, and why the cost of imbalance (×1/π in samples) is a design number, not a footnote.
- **Same structure elsewhere:** fraud (0.17%), AML transactions (0.1%), QEC decoding (0.14% logical errors). Imbalance-weighted sketching carries over unchanged. One sentence in the talk, no more.

---

## 6. Realism: the resource ledger (what the organisers asked for)

Computed by `resources.py` and shown on one slide and in the dashboard:

| Quantity | All genes (N = 2¹⁵ ≈ 32.7k) | Gene pairs (N = 2³⁰ ≈ 1.07×10⁹) |
|---|---|---|
| Index qubits | 15 | 30 |
| + Hadamard ancilla + work ancillas for multi-controlled phase (≈ n) | ≈ 30 logical | **≈ 60 logical**, matching the paper's "< 60 logical qubits" scale |
| Samples per query, M ≈ N·Q²/ε (ε=0.1, Q=1) | ≈ 3.3×10⁵ | ≈ 1.1×10¹⁰ |
| × imbalance factor 1/(π(1−π)), dendritic (π=0.014) | ≈ 2.4×10⁷ | ≈ 7.7×10¹¹ |
| Gates per shot (≈ M multi-controlled phases, ≈ O(n) two-qubit gates each) | ≈ 10⁸–10⁹ | ≈ 10¹³ |
| Classical memory for the same sketch | 32.7k floats (trivial) | ≈ 10⁹ floats (≈ 4–8 GB per class vector), and the lower bound bites only in regimes like this |

**The honest conclusion, which goes on the slide:** this needs **fault-tolerant** logical qubits with fast, MHz-class logical gates, and the advantage is in **memory/size, not time**. At 15 genes' worth of qubits classical wins easily. The argument becomes interesting only at feature spaces of about 10⁹ and more, such as gene-pair interactions and multi-omics. The hardware toy shows the circuit primitive, not usefulness. Judges reward this kind of honesty under "theoretical grounding".

---

## 7. Deliverables

### 7.1 Slides (7 slides, 5 minutes, about 40 s each)

1. **Hook:** "1 in 70 blood cells is a dendritic cell; 1 in a million can be a tumour cell. Can a 60-qubit machine learn them from a stream too big to store?" + pipeline picture.
2. **The data:** PBMC3k table (dimension, 97% sparse, class counts). "Hard because memory ≈ features and the signal is rare."
3. **Mechanism:** one record = one phase kick; the drum analogy; the equation φ_g ≈ c(μ₁−μ₀); Hadamard-test readout.
4. **Why it fits (mechanistic):** log₂N qubits; M=Θ(NQ²/ε); M·S ≥ Ω(N·Q_C); + E2 plot (N/M law).
5. **Our twist: the cost of rare classes:** E3 plot, factor 1/(π(1−π)), measured vs predicted.
6. **It runs:** E4/E5. Simulator → mock → IQM hardware, calibrated scores vs ideal, CZ count. Small dashboard screenshot.
7. **Real world and realism:** resource ledger, claims register in two lines ("memory, not speed; FT needed; classical wins at this scale"), and the next step (gene-pair features, shadows readout as in the thesis).

Backup slides (for Q&A only): baselines table, all-zero-cell handling, calibration fit, label caveat.

### 7.2 Q&A preparation (likely attacks, with answers)

| Question | Answer |
|---|---|
| "Classical does this with 33k floats." | Yes, and we show it. The claim is about the regime N ≫ memory (gene pairs, 10⁹ features), where the paper proves a memory–sample trade-off. |
| "Does your centroid task inherit the lower bound?" | Not proven. The paper's bound is for its task families (NOPE, forrelation-type). We say it is open. |
| "Loading data kills quantum advantages." | Sketching *is* the loading. Each record is one gate and is never stored, so there is no QRAM. The cost is samples ∝ N/ε, which we show. |
| "Hardware noise?" | Calibration circuits measure the shrink factor α. We report raw and calibrated values, and the shallow depth (7 CZ) is the reason the toy works. |
| "Why not train on hardware?" | Against the rules. Training (sketch accumulation) is classical simulation; hardware runs inference with fixed phases. |
| "Is this just a linear model?" | Yes, on purpose. A one-query sketch gives a linear functional. Nonlinear models need more queries Q, and the cost grows as Q², which is the paper's optimal scaling. |

### 7.3 Working model

`qos_cells` package + `scripts/` + `tests/`: `pytest` passes, and `python scripts/04_sim_circuits.py` reproduces the scores from a clean clone in under 2 minutes.

### 7.4 TypeScript dashboard (Friday, time-boxed)

- **Stack:** Vite + React + TypeScript + Recharts (or Chart.js). No backend: it reads `public/results/*.json`, copied from `results/`.
- **Panels (in priority order; stop when time runs out):**
  1. **Resource estimator:** sliders for N (log scale), ε, Q, π → qubits, samples, gates, classical memory. Pure TS formulas from Section 6.
  2. **Sample complexity:** E2 curves.
  3. **Imbalance cost:** E3 curves.
  4. **Hardware vs simulator:** E5 bars with error bars.
- **Defensive:** every JSON has `"schema_version"`. A small runtime check validates fields, and a mismatch shows a **red banner naming the file**, never a blank chart. The page also works offline (`npm run build`, then open `dist/`).
- **Use in the talk:** a screenshot on slide 6. A live demo only if a spare 20 s exists, never as a dependency.

---

## 8. Schedule (time-boxed, with go/no-go gates)

Times are Stockholm. Each gate has a **hard deadline**. If a gate is missed, take the listed fallback, don't push on.

### Thursday 5 Nov

| Time | Lane | Work | Gate / output |
|---|---|---|---|
| 14:00–14:20 | A | Read this plan. Create a venv. `pip install "iqm-client[qiskit]" scanpy scikit-learn matplotlib pytest` (install iqm first so it pins a compatible Qiskit). Set the token: `$env:IQM_TOKEN="..."` (PowerShell) | — |
| 14:20–14:35 | A/C | **Talk to the IQM person now:** confirm token, which machine, whether 3-qubit jobs tonight are fine, and when they are reachable Friday morning. Run `python qpu_pick.py --qubits 3` | **G0 (14:35):** token works, `qpu_pick` lists ≥1 ready machine. *If not:* keep building; hardware moves to the Friday-morning slot with the IQM person; mock is used tonight |
| 14:35–15:45 | A | `load_data.py` + `01_baselines.py` (E1). Stratified split, seed, gene selection on train only, all-zero-cell counter, label counts printed | **G1 (15:45):** E1 JSON exists, numbers near the dry-run values. *If scanpy fails on Windows:* `pip install anndata` and read the 10x matrix from the scanpy URL by hand; or switch to `sc.datasets.pbmc68k_reduced()` (700 cells, 765 genes, already bundled) |
| 15:45–17:15 | A | `sketch.py` + golden test + `02_scaling.py` (E2) + `03_imbalance.py` (E3) | **G2 (17:15):** golden test passes; E2 slope ≈ N/M; E3 penalty within ×2 of 1/(π(1−π)). *If E2/E3 misbehave:* first suspect the phase scaling (wrap-around assert) and weight signs; after 30 min of debugging, present the exact-phase classifier plus the synthetic-only scaling (L2) |
| 17:15–18:30 | A | `circuits.py` + `04_sim_circuits.py` (E4): statevector equality test, shot sim, transpile to IQM, **IQM mock run** via `pick_and_transpile(..., mock=True)` | **G3 (18:30):** mock results within 3σ of shot sim. *If transpile fails:* drop `DiagonalGate`, build N=4 controlled-diag by hand (CP + CZ decomposition) or use 2 qubits (N=2) |
| 18:30–19:15 | A/C | **Hardware attempt 1** (E5, N=4): submit, save job id, freeze phases + hash. If the queue is long, leave it running | Job id on disk |
| 19:15–20:00 | — | Dinner / break (scheduled, not optional; Friday needs a clear head) | — |
| 20:00–21:30 | A (B) | `resources.py` (E6) + `06_figures.py` (all plots from JSON) + **slide skeleton** with placeholder figures. Collect hardware results if they are back | **G4 (21:30):** all 7 slides exist in rough form. This is the L3-safe talk |
| 21:30–22:00 | A | Write one-line notes: what worked, which numbers go on slides. Commit everything (git) and copy to a USB stick / cloud | **Stop at 22:00** |

### Friday 6 Nov

| Time | Lane | Work | Gate / output |
|---|---|---|---|
| 08:30–09:30 | A/C | Hardware results: collect attempt 1 or **run attempt 2 with the IQM person present**. Apply calibration. Write `hw_run.json` | **G5 (09:30):** calibrated hardware scores exist. *If not:* run with the IQM noise model / mock and label it clearly as "emulated, hardware queue failed" (L1). Optional Fujitsu attempt only if G5 passed |
| 09:30–11:30 | B (or A) | **Dashboard:** `npm create vite@latest dashboard -- --template react-ts`, panels 1→4 in order | **G6 (11:30):** panel 1 + one chart working, built to `dist/`. *If not:* screenshot whatever works; the estimator table goes on the slide anyway |
| 11:30–12:30 | A | Final slides with real figures; claims register into speaker notes; backup slides | **Freeze 12:30.** Export PDF + keep PPTX/HTML; two copies (laptop + USB/cloud) |
| 12:30–13:15 | A | Rehearse **three times aloud with a timer** (target 4:30). Cut words, not slides. Practise the 6 Q&A answers | ≤ 5:00 on the third run |
| 13:15–talk | — | Lunch, buffer, check the projector adapter (Windows laptop: HDMI/USB-C) | — |

If the talk turns out to be earlier (for example 13:00), cut in this order: dashboard panels 2–4 → hardware stretch → second rehearsal. Never cut the slide freeze.

---

## 9. When things go wrong

### 9.1 Degradation ladder (the talk exists at every level)

| Level | Condition | What we present |
|---|---|---|
| **L0** | Everything works | Full story: E1–E6, hardware, dashboard |
| **L1** | No real hardware result by 09:30 Fri | Same story; slide 6 shows the IQM mock / noise-model run, labelled clearly. Mention the submitted job id and the reason |
| **L2** | Sketch scaling on real data is unstable | Exact-phase (non-sketched) classifier on real data + sketch scaling on the synthetic benchmark (E2/E3 synthetic) + theory |
| **L3** | Code broadly broken at 10:00 Fri | Theoretical talk (explicitly allowed): mechanism, resource ledger, imbalance derivation (a two-line variance argument), dry-run numbers from this plan clearly labelled as such |

### 9.2 Specific failure modes

| Failure | Detection (loud) | Response |
|---|---|---|
| `iqm-client` / Qiskit version clash on Windows | Import error at 14:20 | Separate venv only for hardware (`iqm-client[qiskit]` alone). The science code needs only NumPy/scanpy. Ask the IQM person for their known-good versions |
| Token missing / expired | `qpu_pick` raises "IQM_TOKEN is not set" | Ask the IQM person; meanwhile mock only |
| No ready machine (`no real machine ... takes jobs now`) | RuntimeError from `pick_and_transpile` | Retry every 30 min; ask about run windows; L1 if nothing by 09:30 |
| Job queued overnight and lost | Job id not saved | Prevented: the id is written to disk before waiting |
| Hardware ⟨Z⟩ ≈ 0 for everything | Calibration α < 0.3 → script raises | Try another machine/qubit layout (`qpu_pick` order); reduce to 2 qubits (N=2); else L1 |
| Silent sign flips | Assert max|φ| ≤ π/2 | Rescale via `rescale_safe` |
| All-zero cells | Counter in loader, abstain bucket | Report the abstain rate on the backup slide; choose genes by |w| (fewer zeros) |
| Data leakage (genes chosen on all data) | Code review checklist: selection inside the train split | Fix before G1 |
| Plain accuracy reported | Lint rule in the figures script: only balanced accuracy / AUPRC keys allowed | — |
| Laptop dies / disk loss | — | git commit + copy to USB/cloud at G4 and at the freeze; slides as PDF on a second device |
| Running out of time | Missed gate | Take the gate's fallback immediately; do not borrow from the slide freeze |
| Q&A attack on "exponential advantage" | — | Claims register: memory, not time; lower bound not proven for our task |

---

## 10. Why this plan (argument against the alternatives)

| Alternative | Why not chosen as the main line |
|---|---|
| Google surface-code data (0.14% logical errors) | Strong property, but parsing the Zenodo format and building a decoder-aware quantum model in 24 h is risky. The mechanistic quantum story is weaker than its classical (AlphaQubit) context |
| `qspin` + symmetry-equivariant circuits | Good mechanistic story (symmetry inductive bias), but it is a crowded idea that many teams will pick, and it does not use the person's main expertise |
| Credit-card fraud | The PCA-anonymised features remove the structure you could argue about; only imbalance is left |
| IBM AML graph | Very good fit (graph + imbalance), but 5M edges and graph-to-circuit design do not fit the time box |
| Generic VQC/kernel on any of these | Explicitly discouraged by the organisers |

**Why scRNA-seq + oracle sketching wins on the judging criteria:**
1. **Theoretical grounding (weighted first):** every claim is a formula with a measured check (log₂N qubits, N/ε samples, M·S lower bound, 1/(π(1−π)) imbalance cost), plus an explicit list of what we do *not* claim.
2. **Novelty:** imbalance-aware sketching and the rare-class cost law are not in the source paper. The resource ledger for gene-pair feature spaces is our own.
3. **Presentation:** one dataset, one mechanism, one twist, one hardware proof-point, one honest ledger. Seven slides fit the story into five minutes.
4. **Feasibility:** the hard parts were dry-run before writing this plan: the data loads, the golden test passes, the circuit equals the formula, and the transpiled depth is 7 CZ.

---

## 11. Checklist before walking on stage

- [ ] `pytest` green; all figures regenerated from `results/*.json` after the last code change
- [ ] Every number on the slides appears in a JSON file (no hand-typed numbers)
- [ ] Hardware slide states: machine name, date, shots, raw vs calibrated, job id, "inference only, fixed parameters"
- [ ] Metrics are balanced accuracy / AUPRC only
- [ ] Realism slide states: qubits, samples, gates, "fault-tolerant needed", "memory not speed"
- [ ] Claims register in speaker notes
- [ ] Talk ≤ 5:00 timed; PDF on two devices; adapter checked
