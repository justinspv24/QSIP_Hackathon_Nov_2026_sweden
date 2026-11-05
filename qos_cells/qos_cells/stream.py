"""The record stream: (cell i, gene g, label y_i, value p_ig) -> (gene index, importance-weighted kick weight).

Importance weighting (the imbalance-aware part)
-----------------------------------------------
A record from the rare class is weighted +1/pi, a record from the common class -1/(1-pi), with pi the
class fraction. The expected kick on gene g is then proportional to the class-mean difference
    E[kick_g]  ∝  mu1_g - mu0_g            (independent of pi: the weighting removes the class-frequency bias)
while the variance of the kick picks up a factor
    Var[kick_g] ∝  E[p^2|1,g]/pi + E[p^2|0,g]/(1-pi)  ≈  q_g / (pi (1 - pi))
so rare classes cost SAMPLES (M), not qubits. Both moments are available in closed form from the pool, so the
sketch error can be predicted with no free parameter (see `expected_phase`, `phase_variance`).

Two sampling modes
------------------
uniform : a record is a uniformly random (cell, gene) pair. Zero entries give a zero kick (no gate), but they
          still count towards M. This is the paper's setting and gives the clean N/M law.
nonzero : a record is a uniformly random NONZERO entry (what a sequencer actually emits: a read of gene g in
          cell i). No wasted records; the expected phase is the same vector up to a global constant.
"""
from __future__ import annotations

import numpy as np


def class_weights(pi: float) -> np.ndarray:
    """w[y] for y in {0, 1}: -1/(1-pi) for the common class, +1/pi for the rare class."""
    if not 0 < pi < 1:
        raise ValueError(f"pi must be in (0,1), got {pi}")
    return np.array([-1.0 / (1.0 - pi), 1.0 / pi])


class RecordStream:
    def __init__(self, p: np.ndarray, y: np.ndarray, pi_hat: float, mode: str = "uniform", abstain=None):
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if abstain is not None:
            p, y = p[~abstain], y[~abstain]
        if p.ndim != 2 or len(y) != p.shape[0]:
            raise ValueError("p must be (cells, N) and y (cells,)")
        self.p, self.y, self.mode = p, y, mode
        self.n_cells, self.N = p.shape
        self.w_class = class_weights(pi_hat)
        self.w_cell = self.w_class[self.y]                     # per-cell importance weight
        if mode == "nonzero":
            self.nz_i, self.nz_g = np.nonzero(p)
            self.population = int(len(self.nz_i))
        elif mode == "uniform":
            self.population = int(self.n_cells * self.N)
        else:
            raise ValueError(f"unknown mode {mode}")
        if self.population == 0:
            raise ValueError("empty stream: no records available")

    # ---- sampling -------------------------------------------------------------------------------------------
    def draw(self, M: int, rng: np.random.Generator):
        """M records -> (gene_idx (M,), weights (M,)). weight = p_ig * w_class[y_i]."""
        if self.mode == "uniform":
            i = rng.integers(0, self.n_cells, M)
            g = rng.integers(0, self.N, M)
        else:
            k = rng.integers(0, self.population, M)
            i, g = self.nz_i[k], self.nz_g[k]
        return g, self.p[i, g] * self.w_cell[i]

    # ---- closed-form moments (for predictions; the quantum computer never needs these) ------------------------
    def expected_phase(self, tau: float) -> np.ndarray:
        """E[phi] after M records (independent of M): tau * (1/population) * sum_i p_ig * w_i."""
        return tau * (self.w_cell @ self.p) / self.population

    def phase_variance(self, tau: float, M: int) -> np.ndarray:
        """Var[phi_g] after M records: (tau^2/M) * [ E[(p w 1[g])^2] - E[p w 1[g]]^2 ]."""
        m1 = (self.w_cell @ self.p) / self.population
        m2 = ((self.w_cell**2) @ (self.p**2)) / self.population
        return (tau**2 / M) * (m2 - m1**2)

    def predicted_relative_error(self, tau: float, M: int) -> float:
        """sqrt(sum_g Var[phi_g]) / ||E[phi]||: the expected relative L2 error of the sketch."""
        target = self.expected_phase(tau)
        return float(np.sqrt(self.phase_variance(tau, M).sum()) / np.linalg.norm(target))

    def class_mean_difference(self) -> np.ndarray:
        """Delta_g = mu1_g - mu0_g on p (what the exact centroid classifier uses)."""
        return self.p[self.y == 1].mean(0) - self.p[self.y == 0].mean(0)


def synthetic_profiles(N: int, rng: np.random.Generator, n_markers: int = 8, marker_boost: float = 6.0):
    """Two expression profiles on the simplex: a common class c0 and a rare class c1 with boosted marker genes."""
    base = rng.gamma(0.7, 1.0, N)
    c0 = base / base.sum()
    boost = np.ones(N)
    boost[rng.choice(N, n_markers, replace=False)] = marker_boost
    c1 = c0 * boost
    return c0, c1 / c1.sum()


def synthetic_pool(c0: np.ndarray, c1: np.ndarray, n_total: int, pi: float, rng: np.random.Generator,
                   reads_per_cell: int = 30):
    """A sparse synthetic 'expression' pool with a controlled class fraction pi.

    Each cell draws `reads_per_cell` reads from a multinomial over the N genes, so p = counts / reads sums to 1 and
    most entries are zero (like scRNA-seq). Returns p (cells, N), y (cells,) and the realised pi.
    """
    n1 = max(2, int(round(n_total * pi)))
    n0 = n_total - n1
    counts0 = rng.multinomial(reads_per_cell, c0, size=n0)
    counts1 = rng.multinomial(reads_per_cell, c1, size=n1)
    p = np.vstack([counts0, counts1]).astype(np.float64) / reads_per_cell
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    return p, y, n1 / n_total
