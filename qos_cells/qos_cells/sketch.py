"""Quantum oracle sketching for a diagonal unitary (Zhao, Zlokapa, Preskill, Huang 2026, arXiv:2604.07639).

Each streamed record (gene g, weight w) is ONE diagonal phase gate on the log2(N)-qubit index register:
    V_t = exp( i * tau * w_t / M * |g_t><g_t| )
All the gates are diagonal, so they commute, and after M records the register carries
    U = diag( exp(i * phi_g) ),    phi_g = (tau / M) * sum_{t : g_t = g} w_t
The record is thrown away after its gate: the register remembers the SUM of the kicks, never the records.

Readout (one query): amplitude-encode a cell |psi> = sum_g a_g |g>, run a Hadamard test on controlled-U with an
S-dagger before the final Hadamard, and the ancilla gives
    <Z_anc> = Im <psi|U|psi> = sum_g a_g^2 sin(phi_g)
which is a linear score in p_g = a_g^2 (the squared amplitudes).

Physical analogy: raindrops on a set of tuned drums. No drum stores the rain, but after a storm each drum is turned
by an angle that reflects how much rain fell on it.
"""
from __future__ import annotations

import numpy as np


def sketch_phases(gene_idx, weights, N: int, M: int, tau: float) -> np.ndarray:
    """Accumulate the phase kicks of M records. Returns phi (N,). This IS the oracle sketch (classically simulated)."""
    gene_idx = np.asarray(gene_idx)
    weights = np.asarray(weights, dtype=np.float64)
    if len(gene_idx) != len(weights):
        raise ValueError("gene_idx and weights must have the same length")
    if len(gene_idx) and (gene_idx.min() < 0 or gene_idx.max() >= N):
        raise ValueError("gene index out of range")
    phi = np.zeros(N)
    np.add.at(phi, gene_idx, tau * weights / M)
    if not np.all(np.isfinite(phi)):
        raise FloatingPointError("non-finite phase")
    return phi


def rescale_safe(phi: np.ndarray, limit: float = np.pi / 2):
    """Scale phi so that max|phi| = limit, which keeps sin() monotone (no silent sign flips).

    Returns (phi_scaled, c). Fails loud on an all-zero sketch (empty stream or missing labels).
    In a real stream the scale tau is fixed in advance from a bound (|p| <= 1) or a pilot batch; here the sketch is
    simulated classically, so the rescaling is applied after the fact and c is reported.
    """
    m = float(np.abs(phi).max())
    if m == 0:
        raise ValueError("sketch is all zero: stream empty or labels missing")
    c = limit / m
    out = phi * c
    assert np.abs(out).max() <= limit + 1e-12, "phase wrap-around: |phi| > limit"
    return out, c


def hadamard_score(a: np.ndarray, phi: np.ndarray) -> float:
    """Ideal <Z_anc> of the Hadamard test (with S-dagger) = sum_g a_g^2 sin(phi_g), for one cell."""
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a)
    if n == 0:
        raise ValueError("all-zero cell on the selected genes -> abstain")
    return float(np.sum((a / n) ** 2 * np.sin(phi)))


def scores_from_p(p: np.ndarray, phi: np.ndarray, abstain=None) -> np.ndarray:
    """Vectorised ideal Hadamard-test scores for many cells (rows of p sum to 1). Abstained cells get -inf."""
    s = p @ np.sin(phi)
    if abstain is not None:
        s = np.where(abstain, -np.inf, s)
    return s


def exact_phases(p: np.ndarray, y: np.ndarray, limit: float = np.pi / 2):
    """The M -> infinity sketch (uniform mode): phi ∝ mu1 - mu0 on p, rescaled. The exact centroid classifier."""
    delta = p[y == 1].mean(0) - p[y == 0].mean(0)
    return rescale_safe(delta, limit)
