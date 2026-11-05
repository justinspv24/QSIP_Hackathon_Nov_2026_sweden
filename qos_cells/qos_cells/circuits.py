"""Qiskit circuits for the one-query readout of the sketched oracle.

    q0        = ancilla (Hadamard test)
    q1 .. qn  = gene index register, n = log2(N)

    |0>_anc |0>_reg  --prepare |psi> = sum_g a_g |g> on reg-->  H(anc)  -->  controlled-diag(e^{i phi_g})  -->  S†(anc) H(anc)  -->  measure anc
    <Z_anc> = Im <psi|U|psi> = sum_g a_g^2 sin(phi_g)

The controlled-diagonal is written as ONE diagonal gate on anc+reg: entry 2g+1 (ancilla = 1, register = g) gets
e^{i phi_g}, entry 2g gets 1. The ancilla is qubit 0, i.e. the least-significant bit of the basis-state index.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector


def hadamard_test_circuit(a, phi, measure: bool = True) -> QuantumCircuit:
    a = np.asarray(a, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    N = len(a)
    n = int(round(np.log2(N)))
    if 2**n != N or len(phi) != N:
        raise ValueError("N must be a power of 2 and len(phi) == N")
    norm = np.linalg.norm(a)
    if norm == 0:
        raise ValueError("all-zero cell: no quantum state -> abstain")
    a = a / norm
    qc = QuantumCircuit(n + 1, 1 if measure else 0)
    qc.prepare_state(list(a), list(range(1, n + 1)))
    qc.h(0)
    d = np.ones(2 ** (n + 1), dtype=complex)
    for g in range(N):
        d[2 * g + 1] = np.exp(1j * phi[g])
    qc.append(DiagonalGate(list(d)), list(range(n + 1)))
    qc.sdg(0)
    qc.h(0)
    if measure:
        qc.measure(0, 0)
    return qc


def ideal_z(a, phi) -> float:
    """<Z_anc> from the exact statevector of the (unmeasured) circuit."""
    qc = hadamard_test_circuit(a, phi, measure=False)
    probs = np.abs(Statevector(qc).data) ** 2
    k = np.arange(len(probs))
    return float(np.sum(probs * (1 - 2 * (k & 1))))


def formula_z(a, phi) -> float:
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    return float(np.sum(a**2 * np.sin(phi)))


def calibration_phases(N: int, n_cal: int, rng: np.random.Generator, limit: float = np.pi / 2):
    """Known-answer circuits for hardware calibration: the all-zero sketch (ideal <Z> = 0) and random sketches
    with max|phi| = limit, so their ideal scores spread over [-1, 1] and the fit z_hw = alpha z_ideal + beta is
    well conditioned. Random sketches, not phi == ±pi/2 everywhere: a constant phase would transpile to a trivial
    circuit and would not exercise the two-qubit gates."""
    out = [np.zeros(N)]
    for _ in range(n_cal - 1):
        v = rng.uniform(-1, 1, N)
        v *= limit / np.abs(v).max()
        out.append(v)
    return out


def run_shots(circuits, shots: int, seed: int = 0, noise_backend=None):
    """Counts from the Aer simulator (ideal), or from a noisy fake backend's own simulator if given."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    if noise_backend is None:
        sim = AerSimulator(seed_simulator=seed)
        tc = transpile(circuits, sim)
        res = sim.run(tc, shots=shots).result()
    else:
        tc = [transpile_iqm(c, noise_backend) for c in circuits]
        res = noise_backend.run(tc, shots=shots, seed_simulator=seed).result()
    return [res.get_counts(i) for i in range(len(circuits))]


def transpile_iqm(circuit, backend, **kw):
    from iqm.qiskit_iqm import transpile_to_IQM

    return transpile_to_IQM(circuit, backend, **kw)


def gate_stats(tc) -> dict:
    ops = {k: int(v) for k, v in tc.count_ops().items()}
    return dict(ops=ops, depth=int(tc.depth()), two_qubit=int(sum(v for k, v in ops.items() if k in ("cz", "move"))),
                n_qubits=int(tc.num_qubits))
