"""Metrics, threshold selection and hardware calibration.

Metrics are balanced accuracy and AUPRC only. Plain accuracy is not computed anywhere: with 0.6 % positives a
classifier that says "common" for everything scores 99.4 % and learns nothing.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score


def finite_scores(s: np.ndarray) -> np.ndarray:
    """Abstained cells (score -inf) are ranked below everything else: predicted negative, never dropped."""
    s = np.asarray(s, dtype=np.float64)
    if np.isneginf(s).all():
        return np.zeros_like(s)
    floor = s[np.isfinite(s)].min() - 1.0
    return np.where(np.isfinite(s), s, floor)


def best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold that maximises balanced accuracy on (train) scores; ties -> the one closest to the midpoint."""
    s = finite_scores(scores)
    u = np.unique(s)
    if len(u) < 2:
        return float(u[0])
    cands = (u[:-1] + u[1:]) / 2
    order = np.argsort(s)
    s_sorted, y_sorted = s[order], y[order]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    cum_pos = np.cumsum(y_sorted)              # positives with score <= s_sorted[k]
    cum_neg = np.cumsum(1 - y_sorted)
    pos_at = np.searchsorted(s_sorted, cands, side="right") - 1
    tpr = 1 - cum_pos[pos_at] / max(n_pos, 1)
    tnr = cum_neg[pos_at] / max(n_neg, 1)
    bacc = (tpr + tnr) / 2
    best = np.flatnonzero(bacc == bacc.max())
    mid = (s[y == 1].mean() + s[y == 0].mean()) / 2
    k = best[np.argmin(np.abs(cands[best] - mid))]
    return float(cands[k])


def evaluate(scores_test: np.ndarray, y_test: np.ndarray, threshold: float) -> dict:
    s = finite_scores(scores_test)
    pred = (s > threshold).astype(int)
    return dict(
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        auprc=float(average_precision_score(y_test, s)),
        recall_pos=float(pred[y_test == 1].mean()) if (y_test == 1).any() else float("nan"),
        recall_neg=float(1 - pred[y_test == 0].mean()),
        n_test=int(len(y_test)), n_test_pos=int(y_test.sum()), base_rate=float(y_test.mean()),
    )


def counts_to_z(counts: dict, n_shots: int | None = None) -> float:
    """<Z> of a single measured bit from a Qiskit counts dict ({'0': n0, '1': n1})."""
    n0 = sum(v for k, v in counts.items() if str(k).replace(" ", "")[-1] == "0")
    n1 = sum(v for k, v in counts.items() if str(k).replace(" ", "")[-1] == "1")
    tot = n0 + n1 if n_shots is None else n_shots
    return (n0 - n1) / tot


def fit_calibration(z_ideal: np.ndarray, z_measured: np.ndarray, alpha_min: float = 0.3) -> dict:
    """Least-squares fit z_measured = alpha * z_ideal + beta. Fails loud if alpha < alpha_min (too noisy for a sign readout)."""
    A = np.c_[np.asarray(z_ideal, float), np.ones(len(z_ideal))]
    (alpha, beta), *_ = np.linalg.lstsq(A, np.asarray(z_measured, float), rcond=None)
    resid = np.asarray(z_measured) - (alpha * np.asarray(z_ideal) + beta)
    cal = dict(alpha=float(alpha), beta=float(beta), rms_residual=float(np.sqrt(np.mean(resid**2))), n=int(len(z_ideal)))
    if alpha < alpha_min:
        raise RuntimeError(f"calibration alpha={alpha:.3f} < {alpha_min}: device too noisy for a sign readout; "
                           "switch qubits or machine")
    return cal


def apply_calibration(z_measured, cal: dict) -> np.ndarray:
    return (np.asarray(z_measured, float) - cal["beta"]) / cal["alpha"]
