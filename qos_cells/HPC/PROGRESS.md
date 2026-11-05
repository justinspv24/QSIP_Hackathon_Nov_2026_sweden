# PROGRESS — E8–E11 on quant1

Start: 2026-11-05 18:42 UTC. Hard target: all results by 22:12 UTC (3.5 h).

## 18:50 — plan, estimates, decisions

| exp | method | est. run | est. done (UTC) |
|---|---|---|---|
| E9 | GPU block math: centroid = K×K Gram of u = x²/‖pair(x)‖; scores = quadratic forms; stream = Poisson-split record sampler (exactly the E7 Poissonised stream, never materialised). Gate at K=512, then K=8192 (N=3.4e7), then all 13,714 genes (N=9.4e7) | 40 min | 20:30 |
| E10 | CELLxGENE Census download (background, starts first), then same GPU pipeline at K=512 | 30 min dl + 25 min run | 21:15 |
| E11 | synthetic Δ̂ = Δ + noise, k = N^α evenly spread; balanced accuracy from the exact Gaussian score law, so N = 2²⁴ needs only vectors, not cells | 15 min | 20:15 |
| E8 | CuPy statevector, in-place H/phase kernels, complex128, n = 10…30 (+1 ancilla; n = 30 uses 32 GB) | 10 min | 20:00 |
| write-up | figures from JSON, README, SLIDE_NOTES | 30 min | 21:45 |

Decisions:
- D1. The node has 8× A6000, not 1. I use GPUs instead of the 64/32-core CPU split: E8 GPU0, E9 GPU1, E10 GPU3, E11 GPU4. CPU is used for data loading and the sklearn gate check.
- D2. SLURM exists, but this node (quant1) is already allocated to us; jobs run with nohup here, no queue wait.
- D3. E9 logreg on all N pair features is done exactly in K×K form (gradient = Uᵀ diag(r) U), same objective as sklearn (class-balanced, C=1, L2). It is gated against sklearn at K=512.
- D4. E11 uses Gaussian test noise and the closed-form balanced accuracy Φ(⟨Δ,D⟩ / 2‖D‖) instead of sampling cells. This is exact for that model and makes N = 2²⁴ cheap.
- D5. Scope cuts made up front: E10 at K=512 only (K=2048 only if time is left); E10 common cells subsampled to ≤ 150k (all rare cells kept); hashing+logreg uses 1 seed except at the biggest N (3 seeds).

## 19:11 — status 1 (T+29 min)

| exp | state | key numbers | ETA |
|---|---|---|---|
| E8 | **done** | circuit = formula to ≤ 3.1e-16 for n = 10…30 index qubits (+1 ancilla); n = 30: 31.5 s, 40.8 GB VRAM; shot noise follows 1/√shots (98–100 % inside 3σ) | — |
| E9 | **done** except 1 job | gate K=512 passes (exact centroid = E7; logreg = E7). S* of the best one-pass sketch stays 8–1,024 up to N = 9.4·10⁷ (27 qubits); S*/N down to 8.5e-8 | logreg at K=13714 (94 M parameters) still fitting: ETA 19:35 or cut |
| E10 | download done (14.2 GB, AIDA, 1,265,624 cells, 625 donors); prep running | — | prep 19:25, runs 19:50 |
| E11 | synthetic **done**; real points added after E10 | Gaussian family: classical S*/N ≤ 2e-3 for α ≤ 0.5, ≥ 0.25 for α ≥ 0.75 | 20:00 |
| whole run | | | ~20:45 (well inside 22:12) |

Decisions since 18:50:
- D6. E8 and E11 were coded by two helper agents in parallel (same repo, same helper module); I reviewed outputs and figures.
- D7. E9 gate: exact centroid and logreg match E7 to 4 decimals. Best one-pass S* per class matches E7 (B, DC: same S*/N; Mk: one grid step). Single-method S* differ by up to 3 grid steps for DC (one DC test cell = 0.045 bal. acc.), from different stream random numbers. Accepted as a pass.
- D8. E9 runs are so fast (K=8192: 4.7 min; K=13714: 6.6 min) that no scope cut was needed: all-genes N = 9.4·10⁷ included, 5 seeds at K ≤ 8192 and 3 seeds at K = 13714, reservoir dropped only at K = 13714. K=2048 rerun with the new code too (E7 values also kept in the JSON).
- D9. E10 dataset: AIDA Phase 1 v2 (CELLxGENE Census 2025-11-08, dataset c838aec3-03ef-4398-b882-0e3912abfff0, sha256 a5bce749…c593). Chosen because it is PBMC (same tissue as PBMC3k), ≥ 200k cells, curated labels, 625 donors, and has rare types down to 1e-4. Tasks: CD1c+ myeloid DC (π 1.0e-2), plasma cell (1.3e-3), hematopoietic stem cell (1.1e-4), CD14+ monocyte (0.16, common).
- D10. E10 uses **all** 1.27 M cells (no subsampling), so every π is the true atlas value. Donor split, 30 % of donors to test.
- D11. Push fails: no GitHub credentials on this node (no gh, no ssh key, https asks for a username). Commits are kept locally on branch `hpc` (2 so far). You need to push from a machine with credentials.

## 19:24 — decisions
- D12. E9 logreg at K=13714 crashed twice (segfault, empty log). Cause: scipy L-BFGS-B workspace (~(2m+5)·n doubles, int32-indexed) overflows at n = 9.4·10⁷ with history m = 10. Fix: m = 5 when n > 5·10⁷ (same objective and optimum, different path). Relaunched 19:20.
- D13. E10 at K=512 runs in ~5–10 min per task, so the optional K=2048 pair space (N = 2,098,176, 22 qubits) is also run, on GPUs 0/6/7.
- D14. E10 convergence rule adds "rel. error < 1" to E7's rule (BA ≥ exact − 0.01): with 3,975 test positives a noisy Δ̂ can already match the balanced accuracy, so this keeps M from being chosen absurdly small. The M grid is extended to 656,100·N and starts one step below the M whose closed-form rel. error is ≤ 2.5; records per seed are capped at 4·10⁹.

## 19:37 — status 2 (T+55 min)

| exp | state | key numbers | ETA |
|---|---|---|---|
| E8 | **done** (committed) | circuit = formula ≤ 3.1e-16, n ≤ 30 | — |
| E9 | **done** (committed) | best one-pass S* = 8–1,024 counters for N = 1.3·10⁵ … 9.4·10⁷; logreg on all N = 9.4·10⁷ pairs: B 0.994 / DC 0.955 / Mk 0.800 (same as K=512) | — |
| E10 K=512 | **done** | AIDA 1,265,624 cells, donor split. Best one-pass S* = 32–512 of N = 131,328 for all 4 tasks (π 0.16 → 1.2e-4); HSC: 32 test positives, exact 0.945, sketch S* = 128. Sample cost: measured M₁₀ / closed form = 0.93–1.04, slope −0.50; M₁₀·π(1−π)/N = 190–900 over a 1,400× range of π → the 1/(π(1−π)) law holds | — |
| E10 K=2048 | running (4 GPUs) | exact BA: CD1c+ DC 0.969, plasma 0.988, HSC 0.904, CD14+ mono 0.963 | 19:55 |
| E11 | synthetic done; real points pending E10 K=2048 | — | 20:00 |
| write-up | README / slide notes in progress | | 20:20 |

Decisions:
- D15. E10 K=2048 first attempt ran out of GPU memory (two jobs on one GPU + full-size copies of the 892k × 2048 matrix in the Gram and logreg code). Fix: row-batched Gram and logreg forward/backward on views (no copies), one task per GPU. Re-verified: centroid, scores and logreg at K=512 unchanged (logreg 0.9944 / 0.9545 / 0.8000). hashing+logreg is off at K=2048 (would hash ~10¹¹ pair entries per seed) — cut.
- D16. E10 HSC (pairs, K=512) hit the 4·10⁹ records/seed cap at M/N = 218,700; it had already converged at M/N = 72,900 (rel. error 0.78), which is used.

## 19:46 — decisions
- D17. E10 K=2048 done for all 4 tasks, but HSC (K=2048) hit the 4·10⁹ records/seed cap before its stream converged (rel. error 2.35). The GPU sampler does ~3·10⁸ records/s, so the cap was too cautious. Rerunning the three capped runs (HSC K=512 and K=2048, plasma cell K=2048) with cap 10¹¹ and time cap 60 min. The capped results are kept in logs/superseded/. ETA 20:10.

## 20:07 — status 3 (T+85 min)

| exp | state | key numbers | ETA |
|---|---|---|---|
| E8 | **done** | — | — |
| E9 | **done** | — | — |
| E10 | 11 of 12 runs final; HSC K=2048 rerun finishing (last 2 methods) | reruns with cap 10¹¹: HSC K=512 converged at M/N 218,700 (rel. err 0.46), best S* 16–64; plasma K=2048 converged at M/N 24,300 (0.49), best S* 128; HSC K=2048 converged at M/N 72,900 (0.78), run at 218,700 | 20:12 |
| E11 | **done** (real points re-collected after E10) | 16 real gene-pair tasks: log d_eff / log N 0.23–0.74, best S*/N ≤ 4e-3 | 20:15 |
| write-up | README, slide notes, figures drafted | | 20:30 |
| whole run | | | ~20:35 (target 22:12) |

## 20:16 — final status (T+94 min)

| exp | state | headline |
|---|---|---|
| E8 | **done** | circuit = formula ≤ 3.1e-16 up to 30 index qubits; correctness check only |
| E9 | **done** | best one-pass S* = 8–1,024 counters for N up to 9.4·10⁷; S* flat in N |
| E10 | **done** (12/12 runs final, incl. optional K=2048) | AIDA 1.27 M cells: 16–512 counters suffice for all 4 tasks (π 0.16 → 1.1e-4); sample cost matches closed form (0.93–1.04) and the 1/(π(1−π)) law |
| E11 | **done** | synthetic failure only for signal spread ≳ N^0.75; 20 real tasks at y = 0.23–0.74, all solved with S*/N ≤ 4e-3 |

Not done / cut (all logged above): qiskit cross-check in E8; reservoir at K=13714 (E9); hashing+logreg at E10 K=2048; E10 hashing+logreg only S ∈ {64,256,1024}, 1 seed.
Push to origin/hpc: **failed** (no GitHub credentials on quant1). All work is committed locally on branch `hpc`.
