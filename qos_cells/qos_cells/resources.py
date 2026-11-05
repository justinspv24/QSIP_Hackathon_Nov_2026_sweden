"""The resource ledger: what oracle sketching would cost at atlas scale, in qubits, samples, gates and seconds.

Formulas (all stated on the slide):
    index qubits          n = ceil(log2 N)
    logical qubits        ≈ n (index) + 1 (Hadamard ancilla) + (n - 1) (work ancillas for an n-controlled phase) ≈ 2n
    samples per oracle    M = N Q^2 / eps           (paper: Theta(N Q^2 / eps) for Q queries to error eps)
    imbalance factor      1 / (pi (1 - pi))         (our addition: importance-weighted kicks, see stream.py)
    gates per sketch      ≈ M * (2 n - 1)           (one n-controlled phase per record ≈ 2n-1 Toffoli-class gates with ancillas)
    time per sketch       gates / logical_gate_rate
    classical memory      N floats for the same class-difference vector (4 N bytes)
    classical lower bound M * S >= Omega(N Q_C)      (paper, NOPE task family; NOT proven for our centroid task — we say so)
"""
from __future__ import annotations

import math


def ledger(N: int, eps: float = 0.1, Q: int = 1, pi: float = 0.014, gate_rate_hz: float = 1e6) -> dict:
    n = math.ceil(math.log2(N))
    logical = 2 * n
    M = N * Q**2 / eps
    imb = 1.0 / (pi * (1 - pi))
    M_imb = M * imb
    gates = M_imb * (2 * n - 1)
    seconds = gates / gate_rate_hz
    return dict(
        N=int(N), index_qubits=n, logical_qubits=logical,
        samples_balanced=M, imbalance_factor=imb, samples_rare=M_imb,
        gates_per_sketch=gates, seconds_per_sketch=seconds, years_per_sketch=seconds / (365.25 * 86400),
        classical_floats=int(N), classical_bytes=int(4 * N),
        eps=eps, Q=Q, pi=pi, gate_rate_hz=gate_rate_hz,
    )


def human(x: float) -> str:
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    return f"{m:.1f}e{e}"
