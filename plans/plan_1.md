# QSIP Hackathon – Plan 1

**Event:** QSIP Hackathon, ENCCS Stockholm, Thu 5 – Fri 6 Nov 2026
**Challenge:** ExtremeData4QML (data-centric QML) – https://github.com/Ilmosal/ExtremeData4QMLHackathon
**Status:** Plan 1, agreed Thu 5 Nov ~13:35. Later plans will be saved as plan_2, plan_3, …

---

## 1. Challenge in short

- Start from the **data**, not the model.
- Pick a dataset with an extreme property, explain why it is hard, argue **mechanistically** which quantum model fits it, build a small prototype (simulator), connect to the real world with realistic qubit/runtime numbers.
- No generic QNN-on-generic-data. Don't chase "beat classical".
- **No training on real hardware** (inference/validation with fixed parameters is fine).
- Deliverable: slides + **5-minute talk**, Friday afternoon. No leaderboard.
- Judged on: reasoning first, then creativity, then presentation.

## 2. Our approach: quantum oracle sketching

Based on Zhao, Zlokapa, Preskill, Huang (2026), "Exponential quantum advantage in processing massive classical data", arXiv:2604.07639. Paper's GitHub code is available.

- Extreme property: **data too big / too fast to store** (massive stream) + **extreme class imbalance**.
- Mechanism: streaming classical samples are turned into coherent quantum queries (no QRAM), read out with classical shadows. Memory scales with log of dimension.

Key scaling:

- Qubits: n = ceil(log2 d)
- Classical streaming memory: O(d); quantum: O(log d)

**One-sentence pitch:** "Some data is too big and too fast to store. Oracle sketching processes it as a stream into a few qubits. We show where that works, and where it breaks."

## 3. Dataset (decided)

**Main: KDD Cup 99 (10% subset)** – network intrusion detection.
- ~500k records (10%), ~4.9M full; 41 features (~120 after one-hot of protocol/service/flag).
- Qubits: ceil(log2 41) = 6; ceil(log2 ~120) = 7.
- 22 attack types in 4 groups (DoS, probe, R2L, U2R); U2R only 52 records in 10% subset → built-in rare-class failure test.
- Load: `sklearn.datasets.fetch_kddcup99()` (one call, no account).
- Task: binary normal vs attack, plus rare-class experiment.

**Backup:** Credit Card Fraud (ULB), 285k rows, 30 features (5 qubits), 0.17% fraud – only if someone already has the Kaggle file.

**Data handling rules:**
- Deduplicate (KDD 99 has many duplicates – judges will ask).
- Metrics: per-class recall + AUPRC, not plain accuracy.
- Hardware toy: 4–8 most informative features → 2–3 qubits.

## 4. Execution: three tiers

| Tier | Question it answers | Size |
|---|---|---|
| 1. Classical streaming baseline | How much memory does classical need for accuracy X? | Full 10% subset, d = 41 / ~120 |
| 2. Simulator (oracle sketching) | Comparable performance with ~log2 d qubits? Where does it break (rare classes)? Also with noise model. | Subsets, d up to ~64–128 (6–7 qubits) |
| 3. IQM hardware (toy) | Does the mechanism survive real noise? | 2–3 qubits, a few streamed samples (T ≈ 4–16) |

**Hardware caveat (state it ourselves in the talk):** each streamed sample adds gates, so circuit depth ∝ T. On hardware only short streams are faithful. Frame as: "mechanism works at small scale on real hardware; here is how error grows with T."

Hardware = inference only, parameters trained in simulator. Compare: IQM vs ideal sim vs noisy sim (noise model from IQM calibration data if available).

## 5. Final products

1. **One working model:** classifier trained in simulator, `predict(x)` → normal/attack via the oracle-sketching pipeline. Same frozen model used for IQM toy run.
2. **Dashboard (TypeScript – not Streamlit), built later.** Replay-style: reads saved result files (CSV/JSON); does NOT call the QPU or run heavy simulation live.
   Panels:
   - Live stream replay: connections arriving, predicted vs true, attack alerts
   - Memory: classical O(d) vs quantum O(log d) as d grows
   - Performance: per-class recall + AUPRC, classical vs quantum
   - Failure regime: rare-attack (U2R/R2L) recall vs imbalance
   - Hardware: IQM vs ideal sim vs noisy sim
   - Rule: **freeze the results file format early** so dashboard and experiments can be built in parallel.
3. **Slides + 5-min talk**, with the dashboard as a ~30-second demo.

## 6. Timeline

| When | Code | Theory / slides |
|---|---|---|
| Thu 13:30–14:00 | Env setup, clone paper code, KDD loader (dedupe, encode, scale) | Talk to IQM person (questions below) |
| Thu 14:00–16:00 | Tier 2: oracle sketching in simulator on small subset | Tier 1: classical streaming baseline + memory accounting |
| **Thu 15:00** | **Go/no-go:** if no simulator output, shrink scope (fewer features, binary) – don't switch topic | |
| Thu 16:00–17:30 | 2–3 qubit toy circuit, verify in sim, **submit IQM jobs today** | Main plot: memory vs d, performance |
| Thu evening | Freeze results format; dashboard skeleton (TypeScript) with dummy data | |
| Fri 9:00–10:00 | Collect IQM results; hardware vs sim vs noisy sim; error vs T; plug real results into dashboard | Rare-class failure-regime plot |
| Fri 10:00–11:30 | Freeze code | Slides |
| Fri afternoon | Present | |

## 7. Questions for the IQM person

- Which device, how many qubits, connectivity, native gates?
- SDK: Qiskit-on-IQM or Cirq-on-IQM? Ready example notebook? (repo has `qpu_pick.py` to check queue/occupancy)
- Shot budget and queue time; can we submit tonight and collect Friday morning?
- Calibration data (gate fidelities, T1/T2) for a matching noisy simulator?

## 8. Slide skeleton (5 min)

1. Data: massive network traffic stream, can't store it; rare attacks
2. Why hard: size + extreme imbalance
3. Mechanism: oracle sketching, memory O(log d) vs O(d)
4. Model: pipeline diagram
5. Results: memory plot + performance (classical vs quantum)
6. Where it breaks: rare classes
7. Real hardware: IQM toy result
8. Real world: intrusion / fraud detection on streams; realistic qubit counts
