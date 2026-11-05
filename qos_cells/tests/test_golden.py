"""Golden tests — nothing else is trusted until these pass."""
import numpy as np
import pytest

from qos_cells.sketch import sketch_phases, rescale_safe, hadamard_score, exact_phases
from qos_cells.stream import RecordStream, class_weights
from qos_cells.circuits import hadamard_test_circuit, ideal_z, formula_z, run_shots
from qos_cells.readout import counts_to_z, best_threshold, fit_calibration, apply_calibration


def test_boolean_oracle_recovered():
    """The paper's phase oracle (-1)^f(x) is recovered from a uniform stream of (x, f(x)) records."""
    rng = np.random.default_rng(0)
    N, M = 16, 200_000
    f = rng.integers(0, 2, N)
    x = rng.integers(0, N, M)
    phi = sketch_phases(x, f[x], N, M, tau=np.pi * N)
    assert np.allclose(np.sign(np.cos(phi)), (-1.0) ** f)


def test_expected_phase_matches_class_mean_difference():
    """Importance-weighted kicks: E[phi] ∝ mu1 - mu0 whatever the class fraction, and the sketch converges to it."""
    rng = np.random.default_rng(1)
    N, n = 8, 3000
    p = rng.dirichlet(np.ones(N) * 0.3, n)
    y = (rng.uniform(size=n) < 0.05).astype(int)
    p[y == 1, :3] *= 3
    p /= p.sum(1, keepdims=True)
    st = RecordStream(p, y, pi_hat=float(y.mean()), mode="uniform")
    target = st.expected_phase(tau=N)
    delta = st.class_mean_difference()
    assert np.allclose(target, delta, atol=1e-12)                       # tau = N -> E[phi] = Delta exactly
    g, w = st.draw(2_000_000, rng)
    phi = sketch_phases(g, w, N, 2_000_000, tau=N)
    rel = np.linalg.norm(phi - target) / np.linalg.norm(target)
    assert rel < 0.15, rel
    pred = st.predicted_relative_error(N, 2_000_000)
    assert 0.3 < rel / pred < 3.0, (rel, pred)                          # closed-form variance prediction is right


def test_nonzero_mode_same_direction():
    rng = np.random.default_rng(2)
    N, n = 8, 2000
    p = rng.dirichlet(np.ones(N) * 0.2, n)
    p[rng.uniform(size=p.shape) < 0.5] = 0
    keep = p.sum(1) > 0
    p, y = p[keep], (rng.uniform(size=keep.sum()) < 0.2).astype(int)
    p[y == 1, :2] += 0.5
    p /= p.sum(1, keepdims=True)
    st_u = RecordStream(p, y, float(y.mean()), "uniform")
    st_z = RecordStream(p, y, float(y.mean()), "nonzero")
    eu, ez = st_u.expected_phase(1.0), st_z.expected_phase(1.0)
    cos = eu @ ez / np.linalg.norm(eu) / np.linalg.norm(ez)
    assert cos > 0.999999                                               # same vector up to a global constant


def test_rescale_and_wraparound_guard():
    phi, c = rescale_safe(np.array([0.1, -2.0, 0.5]))
    assert np.isclose(np.abs(phi).max(), np.pi / 2)
    with pytest.raises(ValueError):
        rescale_safe(np.zeros(3))


def test_circuit_equals_formula():
    """Statevector <Z_anc> of the Hadamard-test circuit == sum a^2 sin(phi), for N = 2, 4, 16."""
    rng = np.random.default_rng(3)
    for N in (2, 4, 16):
        a = rng.uniform(0, 1, N)
        phi = rng.uniform(-np.pi / 2, np.pi / 2, N)
        assert abs(ideal_z(a, phi) - formula_z(a, phi)) < 1e-9
        assert abs(hadamard_score(a, phi) - formula_z(a, phi)) < 1e-12


def test_shots_agree_within_noise():
    rng = np.random.default_rng(4)
    a = rng.uniform(0, 1, 4)
    phi = rng.uniform(-np.pi / 2, np.pi / 2, 4)
    qc = hadamard_test_circuit(a, phi, measure=True)
    counts = run_shots([qc], shots=4000, seed=1)[0]
    z = counts_to_z(counts)
    assert abs(z - formula_z(a, phi)) < 4 / np.sqrt(4000)


def test_threshold_and_calibration():
    s = np.r_[np.linspace(-1, 0, 50), np.linspace(0.2, 1, 10)]
    y = np.r_[np.zeros(50, int), np.ones(10, int)]
    t = best_threshold(s, y)
    assert 0 < t < 0.2
    zi = np.array([-1, -0.5, 0, 0.5, 1.0])
    zm = 0.8 * zi + 0.05
    cal = fit_calibration(zi, zm)
    assert np.isclose(cal["alpha"], 0.8) and np.isclose(cal["beta"], 0.05)
    assert np.allclose(apply_calibration(zm, cal), zi)
    with pytest.raises(RuntimeError):
        fit_calibration(zi, 0.1 * zi)


def test_class_weights():
    w = class_weights(0.25)
    assert np.allclose(w, [-4 / 3, 4.0])
