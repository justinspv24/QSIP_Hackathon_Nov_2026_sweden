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
