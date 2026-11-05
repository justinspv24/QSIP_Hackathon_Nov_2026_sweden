"""E7 / Route 2 — classical memory stress test on gene-pair features.

Question
--------
The quantum oracle sketch keeps ceil(log2 N) qubits for N features. Can a *classical* streaming method with memory
S << N already solve our rare-cell task? If yes, there is no memory advantage on this data. If balanced accuracy
collapses unless S ~ N, that is evidence (not proof) for the paper's claim on real data.

Feature space
-------------
Gene-pair products z_(g,h) = x_g * x_h for g <= h over the top-K genes (chosen on the TRAIN split only, by the same
|mu1 - mu0| rule as load_data.select_genes), amplitude-normalised per cell exactly as load_data.amplitude_features:
    p_(g,h) = z_(g,h)^2 / sum_(g'<=h') z_(g',h')^2 = x_g^2 x_h^2 / sum x_g'^2 x_h'^2 .
N = K(K+1)/2:  K = 512 -> N = 131,328 (18 index qubits);  K = 2048 -> N = 2,098,176 (22 index qubits).

The stream
----------
Same record stream as stream.py in `uniform` mode: a record is a uniformly random (cell, feature) pair of the
training split with the importance weight w_i = +1/pi_hat (rare class) or -1/(1-pi_hat) (common class), value
p_ij, and kick (tau/M) w_i p_ij with tau = N, so that the aggregated vector
    phi_j = (N/M) * sum_{records on feature j} w_i p_ij      has      E[phi] = mu1 - mu0 = Delta.
With M = 10^8..10^10 the records are not drawn one by one. Instead each (cell, feature) entry receives an
independent Poisson(M / (N n_train)) number of records (a Poissonised uniform stream: the same first and second
moments as M uniform records, and the total number of records is Poisson(M) instead of exactly M). Zero entries
give zero kicks (no gate), exactly as in stream.py, so only the nonzero entries are visited. Every memory-limited
method below is a function of this stream only: the linear sketches of a stream equal the sketch of its aggregate
(sketches commute with sums), so they are applied to phi; the two non-linear baselines (heavy hitters, reservoir)
are simulated with their exact streaming semantics (see their docstrings).

Memory-limited methods (S = number of counters kept while streaming; bytes count every float, int and hash seed)
----------------------------------------------------------------------------------------------------------------
count_sketch      r = 3 rows of S/3 signed buckets, decode Delta_j = median_r( s_r(j) C_r[h_r(j)] )
hashing           1 row of S signed buckets (feature hashing); decode Delta_j = s(j) C[h(j)]; score = <C, hash(p)>
hashing_logreg    class-balanced logistic regression on the S hashed features (NOT one-pass: needs the n x S matrix)
jl_sparse         sparse Johnson-Lindenstrauss (Kane-Nelson): each feature -> 4 signed rows / sqrt(4), S rows
jl_dense          dense Gaussian JL (S x N matrix from one seed) — only where the matrix fits (K = 512, S <= 1024)
heavy_hitters_mg  Misra-Gries with S/2 counters for the positive-valued and S/2 for the negative-valued kicks,
                  computed through the mergeable-summary property (add all counts, subtract the (k+1)-th largest,
                  keep the positive ones: a valid MG summary of the whole stream); decode = f+ - f-
heavy_hitters_top oracle top-S: keep the S features with the largest |phi_j| exactly. An UPPER bound on any
                  heavy-hitter scheme with S counters (SpaceSaving, MG, ...)
reservoir         S nonzero records kept uniformly at random (= S records drawn uniformly from the nonzero stream);
                  decode = sum of the kept kicks
References: exact centroid Delta on the training split (= the M -> infinity sketch, memory N floats),
stream_full = phi itself (memory N floats, same records), class-balanced logistic regression on all N features,
and the simulated quantum readout p . sin(c phi) with the phases rescaled to |phi| <= pi/2 (sketch.rescale_safe).

Control
-------
A synthetic task with NO structure at the same N: x_ij = (2y_i - 1) s_j delta / 2 + eps_ij, eps ~ N(0,1), random
signs s_j, so the class-mean difference delta * s is spread evenly over all N features (the paper's worst-case
flavour: every feature carries the same tiny signal). delta is set so that the exact centroid estimated from the
training cells reaches a target SNR (balanced accuracy ~ 0.98-0.99).

Outputs: results/e7_route2.json (schema_version 1, config included) and, with `figure`, figures/fig_e7_memory.png
built from the JSON only.

Usage
-----
    python scripts/09_route2_classical_memory.py run [--K 512 2048] [--seeds 5] [--quick]
    python scripts/09_route2_classical_memory.py figure
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qos_cells import ROOT, load_config  # noqa: E402

RESULT_NAME = "e7_route2.json"
FIG_NAME = "fig_e7_memory.png"
POSITIVES = ["B cells", "Dendritic cells", "Megakaryocytes"]
F64, I32 = 8, 4            # bytes per float64 counter / int32 index
SEED_BYTES = 8             # one 64-bit hash seed


def log(msg):
    print(f"[e7 {dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ===================================================================================================================
# hashing utilities (vectorised 64-bit mixing; numpy uint64 arithmetic wraps mod 2^64)
# ===================================================================================================================
_C1, _C2, _C3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(j, seed):
    x = (np.asarray(j, dtype=np.uint64) ^ np.uint64(seed)) * _C1
    x ^= x >> np.uint64(30)
    x *= _C2
    x ^= x >> np.uint64(27)
    x *= _C3
    x ^= x >> np.uint64(31)
    return x


def hbucket(j, seed, B):
    return (_mix(j, seed) % np.uint64(B)).astype(np.int64)


def hsign(j, seed):
    return 1.0 - 2.0 * (_mix(j, seed) >> np.uint64(63)).astype(np.float64)


# ===================================================================================================================
# pair features
# ===================================================================================================================
def pair_features_csr(Xsel):
    """Xsel: sparse (cells x K) log-expression on the K selected genes -> csr (cells x N) of the amplitude features
    p_(g,h) = x_g^2 x_h^2 / sum_(g'<=h') x_g'^2 x_h'^2 over the pairs g <= h, float32 values, int32 indices.
    Feature index j(g,h) = g*K - g(g-1)/2 + (h-g) (row-major upper triangle including the diagonal)."""
    import scipy.sparse as sp

    Xsel = sp.csr_matrix(Xsel)
    Xsel.sort_indices()
    n, K = Xsel.shape
    N = K * (K + 1) // 2
    m = np.diff(Xsel.indptr)
    nnz_out = (m.astype(np.int64) * (m + 1) // 2)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(nnz_out)
    indices = np.empty(indptr[-1], dtype=np.int32)
    data = np.empty(indptr[-1], dtype=np.float32)
    abstain = m == 0
    for i in range(n):
        if m[i] == 0:
            continue
        idx = Xsel.indices[Xsel.indptr[i]:Xsel.indptr[i + 1]].astype(np.int64)
        s = Xsel.data[Xsel.indptr[i]:Xsel.indptr[i + 1]].astype(np.float64) ** 2
        a, b = np.triu_indices(m[i])
        g, h = idx[a], idx[b]
        j = g * K - g * (g - 1) // 2 + (h - g)
        v = s[a] * s[b]
        v /= v.sum()
        indices[indptr[i]:indptr[i + 1]] = j
        data[indptr[i]:indptr[i + 1]] = v
    P = sp.csr_matrix((data, indices, indptr), shape=(n, N))
    P.sort_indices()
    return P, abstain


# ===================================================================================================================
# the stream
# ===================================================================================================================
def poisson_stream(P, w_cell, N, M, rng, chunk=20_000_000):
    """Poissonised uniform record stream over the entries of P (csr or dense, cells x N) with per-cell importance
    weights w_cell. Returns the aggregated kick vector phi (N,) = (N/M) sum c_ij w_i p_ij with c_ij ~ Poisson(M/(N n)),
    plus the separate positive-valued and negative-valued totals (for Misra-Gries) and the realised record count."""
    import scipy.sparse as sp

    n = P.shape[0]
    lam = M / (N * n)
    phi_pos = np.zeros(N)
    phi_neg = np.zeros(N)
    n_records = 0
    if sp.issparse(P):
        P = sp.csr_matrix(P)
        for a in range(0, P.nnz, chunk):
            b = min(a + chunk, P.nnz)
            c = rng.poisson(lam, size=b - a)
            n_records += int(c.sum())
            row_of = np.searchsorted(P.indptr, np.arange(a, b), side="right") - 1      # row of each entry, per chunk
            v = c * (w_cell[row_of] * P.data[a:b].astype(np.float64))
            idx = P.indices[a:b]
            pos = v > 0
            phi_pos += np.bincount(idx[pos], weights=v[pos], minlength=N)
            phi_neg += np.bincount(idx[~pos], weights=-v[~pos], minlength=N)
    else:
        rows = max(1, chunk // N)
        for a in range(0, n, rows):
            b = min(a + rows, n)
            c = rng.poisson(lam, size=(b - a, N))
            n_records += int(c.sum())
            v = (c * (w_cell[a:b, None] * P[a:b].astype(np.float64)))
            s = v.sum(0)
            phi_pos += np.where(s > 0, s, 0.0)          # per-feature net totals of the chunk, split by sign
            phi_neg += np.where(s < 0, -s, 0.0)
    scale = N / M
    return scale * (phi_pos - phi_neg), scale * phi_pos, scale * phi_neg, n_records


def draw_reservoir(P, w_cell, S, rng):
    """S records drawn uniformly from the NONZERO records of the stream (the distribution of a reservoir sample of
    size S). Returns the decoded vector: the sum of the kept kicks (any global scale is irrelevant to the classifier)."""
    import scipy.sparse as sp

    n, N = P.shape
    out = np.zeros(N)
    if sp.issparse(P):
        P = sp.csr_matrix(P)
        k = rng.integers(0, P.nnz, S)
        row_of = np.searchsorted(P.indptr, k, side="right") - 1
        np.add.at(out, P.indices[k], w_cell[row_of] * P.data[k].astype(np.float64))
    else:
        i = rng.integers(0, n, S)
        j = rng.integers(0, N, S)
        np.add.at(out, j, w_cell[i] * P[i, j].astype(np.float64))
    return out


# ===================================================================================================================
# memory-limited sketches: phi (N,) -> decoded vector (N,) + memory ledger
# ===================================================================================================================
def ledger(floats=0, ints=0, seeds=0):
    return dict(floats=int(floats), ints=int(ints), seeds=int(seeds),
                bytes=int(F64 * floats + I32 * ints + SEED_BYTES * seeds))


def sk_count_sketch(phi, S, seeds, rows=3):
    N = len(phi)
    B = max(1, S // rows)
    j = np.arange(N)
    est = np.empty((rows, N))
    for r in range(rows):
        h, s = hbucket(j, seeds[2 * r], B), hsign(j, seeds[2 * r + 1])
        C = np.bincount(h, weights=s * phi, minlength=B)
        est[r] = s * C[h]
    return np.median(est, axis=0), ledger(floats=B * rows, seeds=2 * rows)


def sk_hashing(phi, S, seeds):
    N = len(phi)
    j = np.arange(N)
    h, s = hbucket(j, seeds[0], S), hsign(j, seeds[1])
    C = np.bincount(h, weights=s * phi, minlength=S)
    return s * C[h], ledger(floats=S, seeds=2)


def sk_jl_sparse(phi, S, seeds, s_rows=4):
    N = len(phi)
    j = np.arange(N)
    C = np.zeros(S)
    H, Sg = [], []
    for k in range(s_rows):
        h, s = hbucket(j, seeds[2 * k], S), hsign(j, seeds[2 * k + 1])
        C += np.bincount(h, weights=s * phi, minlength=S)
        H.append(h)
        Sg.append(s)
    dec = np.zeros(N)
    for h, s in zip(H, Sg):
        dec += s * C[h]
    return dec / s_rows, ledger(floats=S, seeds=2 * s_rows)        # R^T R phi with R = (signs/sqrt(s)) -> /s


def sk_jl_dense(phi, S, seed):
    rng = np.random.default_rng(seed)
    N = len(phi)
    R = rng.standard_normal((S, N), dtype=np.float32) / np.sqrt(S)
    C = R @ phi.astype(np.float32)
    return (R.T @ C).astype(np.float64), ledger(floats=S, seeds=1)


def sk_heavy_hitters_mg(phi_pos, phi_neg, S):
    """Misra-Gries with k = S/2 counters per sign class via the mergeable-summary rule."""
    def mg(A, k):
        if k >= len(A):
            return A.copy()
        thr = np.partition(A, len(A) - k - 1)[len(A) - k - 1]      # (k+1)-th largest value
        return np.maximum(A - thr, 0.0)
    k = max(1, S // 2)
    f = mg(phi_pos, k) - mg(phi_neg, k)
    return f, ledger(floats=2 * k, ints=2 * k)


def sk_random_subset(phi, S, seed):
    """Keep exact counters for S features chosen uniformly at random before the stream starts (the index set is
    regenerated from one seed, so it costs S floats + 1 seed). The fair baseline when the signal is dense but
    redundant: no hash collisions, just fewer features."""
    N = len(phi)
    if S >= N:
        return phi.copy(), ledger(floats=N)
    keep = np.random.default_rng(int(seed)).choice(N, S, replace=False)
    out = np.zeros(N)
    out[keep] = phi[keep]
    return out, ledger(floats=S, seeds=1)


def sk_heavy_hitters_top(phi, S):
    N = len(phi)
    if S >= N:
        return phi.copy(), ledger(floats=N, ints=N)
    keep = np.argpartition(-np.abs(phi), S - 1)[:S]
    out = np.zeros(N)
    out[keep] = phi[keep]
    return out, ledger(floats=S, ints=S)


# ===================================================================================================================
# evaluation
# ===================================================================================================================
def evaluate_columns(P_thr, y_thr, P_test, y_test, D, abstain_thr=None, abstain_test=None):
    """Score every column of D (N x m) on the threshold set (train by default) and on test; threshold chosen on the
    former; balanced accuracy + AUPRC on test."""
    from qos_cells.readout import best_threshold, evaluate

    D32 = np.asarray(D, dtype=np.float32)
    s_tr = np.asarray(P_thr @ D32, dtype=np.float64)
    s_te = np.asarray(P_test @ D32, dtype=np.float64)
    out = []
    for c in range(D32.shape[1]):
        a, b = s_tr[:, c].copy(), s_te[:, c].copy()
        if abstain_thr is not None:
            a[abstain_thr] = -np.inf
        if abstain_test is not None:
            b[abstain_test] = -np.inf
        thr = best_threshold(a, y_thr)
        ev = evaluate(b, y_test, thr)
        out.append(dict(balanced_accuracy=ev["balanced_accuracy"], auprc=ev["auprc"]))
    return out


def logreg_balanced(Xtr, ytr, Xte, yte, scale, max_iter=300, C=1.0):
    """Class-balanced L2 logistic regression; features multiplied by a global scale (p values are ~1e-5).
    Sparse inputs are scaled in place (and restored) to avoid a second copy of a 2 GB matrix."""
    import scipy.sparse as sp
    from sklearn.linear_model import LogisticRegression
    from qos_cells.readout import best_threshold, evaluate

    def scaled(X):
        if sp.issparse(X):
            X.data *= np.float32(scale)
            return X
        return X * scale

    def unscale(X):
        if sp.issparse(X):
            X.data /= np.float32(scale)
    clf = LogisticRegression(class_weight="balanced", C=C, max_iter=max_iter, tol=1e-4)
    try:
        Xs = scaled(Xtr)
        clf.fit(Xs, ytr)
        s_tr = clf.decision_function(Xs)
    finally:
        unscale(Xtr)
    try:
        s_te = clf.decision_function(scaled(Xte))
    finally:
        unscale(Xte)
    thr = best_threshold(s_tr, ytr)
    ev = evaluate(s_te, yte, thr)
    return dict(balanced_accuracy=ev["balanced_accuracy"], auprc=ev["auprc"], n_iter=int(np.max(clf.n_iter_)),
                converged=bool(np.max(clf.n_iter_) < max_iter))


def summarise(rows_by_seed):
    """rows_by_seed: list over seeds of dicts {S: metrics}. -> list of rows with mean/std/per-seed."""
    Ss = sorted(rows_by_seed[0].keys())
    rows = []
    for S in Ss:
        ba = np.array([r[S]["balanced_accuracy"] for r in rows_by_seed])
        ap = np.array([r[S]["auprc"] for r in rows_by_seed])
        rows.append(dict(S=int(S), **rows_by_seed[0][S]["mem"],
                         balanced_accuracy_mean=float(ba.mean()), balanced_accuracy_std=float(ba.std()),
                         auprc_mean=float(ap.mean()), auprc_std=float(ap.std()),
                         balanced_accuracy_seeds=[round(float(v), 4) for v in ba],
                         auprc_seeds=[round(float(v), 4) for v in ap]))
    return rows


def s_star(rows, target, key="balanced_accuracy_mean", tol=0.01):
    """Smallest S whose mean score is within tol of the target; None if never reached."""
    for r in rows:
        if r[key] >= target - tol:
            return r["S"], r["bytes"]
    return None, None


def s_star_stable(rows, target, key="balanced_accuracy_mean", tol=0.01):
    """Smallest S such that this and every larger S are within tol of the target (guards against lucky points)."""
    ok = [r[key] >= target - tol for r in rows]
    for i in range(len(rows)):
        if all(ok[i:]):
            return rows[i]["S"], rows[i]["bytes"]
    return None, None


# ===================================================================================================================
# one task (real or synthetic): references, M selection, all methods over the S grid
# ===================================================================================================================
def run_task(name, P_train, y_train, P_test, y_test, pi_hat, N, S_grid, M_over_N_grid, n_seeds, base_seed,
             abstain_train=None, abstain_test=None, do_logreg=True, do_jl_dense=False, hash_logreg_max_S=16384,
             logreg_max_iter=300, quick=False, P_thr=None, y_thr=None, extra_methods=()):
    """P_thr/y_thr: the set the decision threshold is chosen on. Default = the training split (as in E1/E2).
    The synthetic control passes a fresh validation set: with N >> n and no structure the training scores of a
    centroid fitted on the same cells are self-inflated, so a train threshold does not transfer (AUPRC stays
    perfect while balanced accuracy drops); real data has structure and does not show this."""
    import scipy.sparse as sp
    from qos_cells.stream import class_weights
    from qos_cells.sketch import rescale_safe

    t0 = time.time()
    n_train, n_test = P_train.shape[0], P_test.shape[0]
    w_cell = class_weights(pi_hat)[y_train]
    sparse = sp.issparse(P_train)
    if P_thr is None:
        P_thr, y_thr, abstain_thr = P_train, y_train, abstain_train
    else:
        abstain_thr = None
    res = dict(task=name, N=int(N), n_train=int(n_train), n_test=int(n_test), n_train_pos=int(y_train.sum()),
               n_test_pos=int(y_test.sum()), pi_hat=float(pi_hat), base_rate_test=float(y_test.mean()),
               abstain_rate_test=float(abstain_test.mean()) if abstain_test is not None else 0.0)
    if sparse:
        res["train_density"] = float(P_train.nnz / (n_train * N))
        res["nnz_train"] = int(P_train.nnz)
        res["nnz_test"] = int(P_test.nnz)
    else:
        res["train_density"] = 1.0

    # ---- references --------------------------------------------------------------------------------------------
    if sparse:
        ind1 = (y_train == 1).astype(np.float32) / max(int((y_train == 1).sum()), 1)
        ind0 = (y_train == 0).astype(np.float32) / max(int((y_train == 0).sum()), 1)
        delta = np.asarray(P_train.T @ ind1, dtype=np.float64) - np.asarray(P_train.T @ ind0, dtype=np.float64)
    else:
        ind1 = (y_train == 1).astype(np.float32) / max(int((y_train == 1).sum()), 1)
        ind0 = (y_train == 0).astype(np.float32) / max(int((y_train == 0).sum()), 1)
        delta = (ind1 @ P_train).astype(np.float64) - (ind0 @ P_train).astype(np.float64)
    ev = evaluate_columns(P_thr, y_thr, P_test, y_test, delta[:, None], abstain_thr, abstain_test)[0]
    res["exact_centroid"] = dict(**ev, **ledger(floats=N))
    exact_ba, exact_ap = ev["balanced_accuracy"], ev["auprc"]
    log(f"{name}: exact centroid bal.acc {exact_ba:.3f} auprc {exact_ap:.3f}  (N={N:,}, n_train={n_train}, pos={int(y_train.sum())})")
    if do_logreg:
        scale = 1.0 / float(P_train.data.mean()) if sparse else 1.0
        t = time.time()
        res["logreg_full"] = dict(**logreg_balanced(P_train, y_train, P_test, y_test, scale, max_iter=logreg_max_iter),
                                  **ledger(floats=N + 1), feature_scale=scale, seconds=round(time.time() - t, 1))
        log(f"{name}: logreg on all N features bal.acc {res['logreg_full']['balanced_accuracy']:.3f} "
            f"auprc {res['logreg_full']['auprc']:.3f} ({res['logreg_full']['seconds']} s, iters {res['logreg_full']['n_iter']})")

    # ---- M selection: full-memory stream estimate vs M ----------------------------------------------------------
    rngs = [np.random.default_rng(np.random.SeedSequence([base_seed, 7, s])) for s in range(n_seeds)]
    phis = {}
    msel = []
    for mo in M_over_N_grid:
        M = int(mo * N)
        cols, extra = [], []
        for s in range(n_seeds):
            phi, ppos, pneg, nrec = poisson_stream(P_train, w_cell, N, M, rngs[s])
            del pneg                                             # = ppos - phi; not stored (memory)
            phis[(mo, s)] = (phi, ppos, nrec)
            cols.append(phi)
            if sparse:
                ps, c = rescale_safe(phi)
                extra.append(np.sin(ps))
        evs = evaluate_columns(P_thr, y_thr, P_test, y_test, np.stack(cols, 1), abstain_thr, abstain_test)
        row = dict(M_over_N=mo, M=M, nonzero_records_mean=float(np.mean([phis[(mo, s)][2] for s in range(n_seeds)])),
                   balanced_accuracy_mean=float(np.mean([e["balanced_accuracy"] for e in evs])),
                   balanced_accuracy_std=float(np.std([e["balanced_accuracy"] for e in evs])),
                   auprc_mean=float(np.mean([e["auprc"] for e in evs])), auprc_std=float(np.std([e["auprc"] for e in evs])),
                   balanced_accuracy_seeds=[round(e["balanced_accuracy"], 4) for e in evs],
                   rel_error_mean=float(np.mean([np.linalg.norm(c - delta) / np.linalg.norm(delta) for c in cols])))
        if extra:
            q = evaluate_columns(P_thr, y_thr, P_test, y_test, np.stack(extra, 1), abstain_thr, abstain_test)
            row["quantum_sin_readout_balanced_accuracy_mean"] = float(np.mean([e["balanced_accuracy"] for e in q]))
            row["quantum_sin_readout_auprc_mean"] = float(np.mean([e["auprc"] for e in q]))
        msel.append(row)
        log(f"{name}: M/N={mo:>6} -> stream_full bal.acc {row['balanced_accuracy_mean']:.3f}±{row['balanced_accuracy_std']:.3f} "
            f"rel.err {row['rel_error_mean']:.3f} ({row['nonzero_records_mean']:.2e} nonzero records)")
    res["M_selection"] = msel
    conv = [r for r in msel if r["balanced_accuracy_mean"] >= exact_ba - 0.01]
    if conv:
        i = msel.index(conv[0])
        run_row = msel[min(i + 1, len(msel) - 1)]          # one grid step beyond the first converged M
        res["M_converged_over_N"] = conv[0]["M_over_N"]
    else:
        run_row = msel[-1]
        res["M_converged_over_N"] = None
    mo_run = run_row["M_over_N"]
    res["M_run"] = run_row["M"]
    res["M_run_over_N"] = mo_run
    res["stream_full"] = dict(balanced_accuracy=run_row["balanced_accuracy_mean"], auprc=run_row["auprc_mean"],
                              balanced_accuracy_std=run_row["balanced_accuracy_std"], **ledger(floats=N),
                              nonzero_records_mean=run_row["nonzero_records_mean"])
    if "quantum_sin_readout_balanced_accuracy_mean" in run_row:
        res["quantum_sim"] = dict(index_qubits=int(math.ceil(math.log2(N))), logical_qubits=2 * int(math.ceil(math.log2(N))),
                                  balanced_accuracy=run_row["quantum_sin_readout_balanced_accuracy_mean"],
                                  auprc=run_row["quantum_sin_readout_auprc_mean"], M=run_row["M"],
                                  note="classical simulation of the oracle sketch (phases phi, readout p.sin(c phi)); NOT run on hardware at this N")
    log(f"{name}: M_run = {mo_run} N = {res['M_run']:.2e} (first converged M/N = {res['M_converged_over_N']})")
    phis = {s: phis[(mo_run, s)] for s in range(n_seeds)}       # drop the other M values (memory)
    del cols, extra

    # ---- memory-limited methods over the S grid ---------------------------------------------------------------
    methods = {}
    specs = [("count_sketch", "count-sketch (3 rows, median)"), ("hashing", "feature hashing (signed, 1 row)"),
             ("jl_sparse", "sparse JL projection (4 rows)"), ("heavy_hitters_mg", "heavy hitters: Misra–Gries (S/2 + S/2)"),
             ("heavy_hitters_top", "heavy hitters: oracle top-S |Δ|"), ("reservoir", "reservoir: S nonzero records")]
    if do_jl_dense:
        specs.append(("jl_dense", "dense Gaussian JL projection"))
    if "random_subset" in extra_methods:
        specs.append(("random_subset", "random feature subset (S exact counters)"))
    if do_logreg:
        specs.append(("hashing_logreg", "feature hashing + logistic regression"))
    for key, label in specs:
        t = time.time()
        per_seed = []
        for s in range(n_seeds):
            phi, ppos, _ = phis[s]
            pneg = ppos - phi
            ss = np.random.SeedSequence([base_seed, 11, s, specs.index((key, label))])
            rng = np.random.default_rng(ss)
            seeds = rng.integers(0, 2**62, size=16)
            grid = [S for S in S_grid if (key != "jl_dense" or S <= 1024) and (key != "hashing_logreg" or S <= hash_logreg_max_S)]
            if quick:
                grid = grid[::2] + ([grid[-1]] if grid[-1] not in grid[::2] else [])
            cols, mems = [], []
            metrics = {}
            if key == "hashing_logreg":
                scale = 1.0 / float(P_train.data.mean()) if sparse else 1.0
                j = np.arange(N)
                for S in grid:
                    h, sg = hbucket(j, seeds[0], S), hsign(j, seeds[1])
                    Hm = sp.csr_matrix((sg.astype(np.float32), (j, h)), shape=(N, S))
                    Htr = np.asarray((P_train @ Hm).todense() if sparse else P_train @ Hm.toarray())
                    Hte = np.asarray((P_test @ Hm).todense() if sparse else P_test @ Hm.toarray())
                    ev = logreg_balanced(Htr, y_train, Hte, y_test, scale, max_iter=logreg_max_iter)
                    metrics[S] = dict(balanced_accuracy=ev["balanced_accuracy"], auprc=ev["auprc"], mem=ledger(floats=S + 1, seeds=2))
                per_seed.append(metrics)
                continue
            for S in grid:
                if key == "count_sketch":
                    d, m = sk_count_sketch(phi, S, seeds)
                elif key == "hashing":
                    d, m = sk_hashing(phi, S, seeds)
                elif key == "jl_sparse":
                    d, m = sk_jl_sparse(phi, S, seeds)
                elif key == "jl_dense":
                    d, m = sk_jl_dense(phi, S, int(seeds[0]))
                elif key == "heavy_hitters_mg":
                    d, m = sk_heavy_hitters_mg(ppos, pneg, S)
                elif key == "heavy_hitters_top":
                    d, m = sk_heavy_hitters_top(phi, S)
                elif key == "random_subset":
                    d, m = sk_random_subset(phi, S, seeds[0])
                elif key == "reservoir":
                    d, m = draw_reservoir(P_train, w_cell, S, rng), ledger(floats=S, ints=S)
                cols.append(d)
                mems.append(m)
            # batch the scoring: one pass over the data per (method, seed)
            for a in range(0, len(cols), 12):
                evs = evaluate_columns(P_thr, y_thr, P_test, y_test, np.stack(cols[a:a + 12], 1), abstain_thr, abstain_test)
                for S, m, e in zip(grid[a:a + 12], mems[a:a + 12], evs):
                    metrics[S] = dict(balanced_accuracy=e["balanced_accuracy"], auprc=e["auprc"], mem=m)
            per_seed.append(metrics)
        rows = summarise(per_seed)
        Sst, Bst = s_star(rows, exact_ba)
        Sstab, Bstab = s_star_stable(rows, exact_ba)
        Sap, _ = s_star(rows, exact_ap, key="auprc_mean")
        methods[key] = dict(label=label, rows=rows, S_star=Sst, S_star_over_N=(Sst / N if Sst else None), bytes_at_S_star=Bst,
                            S_star_stable=Sstab, S_star_stable_over_N=(Sstab / N if Sstab else None),
                            S_star_auprc=Sap, S_star_auprc_over_N=(Sap / N if Sap else None), seconds=round(time.time() - t, 1))
        log(f"{name}: {key:<18} S* = {Sst} (S*/N = {methods[key]['S_star_over_N']})  stable S* = {Sstab}  [{methods[key]['seconds']} s]")
    res["methods"] = methods
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ===================================================================================================================
# data builders
# ===================================================================================================================
def build_real_task(X, genes, labels, positive, K, seed, test_frac, min_cells):
    from qos_cells.load_data import select_genes, stratified_split

    y = (labels == positive).astype(int)
    tr, te = stratified_split(y, test_frac, seed)
    g = select_genes(X[tr], y[tr], K, min_cells)           # TRAIN ONLY
    P_tr, ab_tr = pair_features_csr(X[tr][:, g])
    P_te, ab_te = pair_features_csr(X[te][:, g])
    return dict(P_train=P_tr, y_train=y[tr], P_test=P_te, y_test=y[te], pi_hat=float(y[tr].mean()),
                abstain_train=ab_tr, abstain_test=ab_te, genes=[str(s) for s in genes[g]])


def build_synthetic(N, n_train, n_test, pi, snr, seed):
    """No-structure control. Returns dense float32 matrices and the parameters used."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99, N]))
    s = rng.choice(np.array([-1.0, 1.0]), size=N)
    n1_tr, n1_te = max(2, int(round(pi * n_train))), max(2, int(round(pi * n_test)))
    tau2 = 1.0 / n1_tr + 1.0 / (n_train - n1_tr)            # per-feature variance of the estimated centroid
    a = 4 * snr**2 / N
    delta2 = (a + math.sqrt(a * a + 4 * a * tau2)) / 2        # (N d^2/2)^2 = snr^2 (N d^2 + N tau^2)
    delta = math.sqrt(delta2)

    def make(n, n1):
        y = np.r_[np.zeros(n - n1, int), np.ones(n1, int)]
        X = rng.standard_normal((n, N), dtype=np.float32)
        X += ((2 * y - 1)[:, None] * (delta / 2) * s[None, :]).astype(np.float32)
        return X, y
    Xtr, ytr = make(n_train, n1_tr)
    Xte, yte = make(n_test, n1_te)
    Xva, yva = make(n_test, n1_te)              # fresh validation cells: the decision threshold is chosen here
    return dict(P_train=Xtr, y_train=ytr, P_test=Xte, y_test=yte, P_thr=Xva, y_thr=yva, pi_hat=float(ytr.mean()),
                params=dict(delta=delta, snr_target=snr, tau=math.sqrt(tau2), noise_sigma=1.0, n_train=n_train, n_test=n_test,
                            n_val=n_test, n_train_pos=n1_tr, n_test_pos=n1_te, pi=pi,
                            threshold_set="separate validation cells (train thresholds do not transfer when N >> n)"))


# ===================================================================================================================
# main run
# ===================================================================================================================
def run(args):
    from qos_cells.load_data import load_pbmc3k
    from qos_cells.resources import ledger as qledger

    cfg = load_config()
    seed = int(cfg["seed"])
    n_seeds = int(args.seeds)
    route2 = dict(K_values=[int(k) for k in args.K], seeds=n_seeds, split_seed=seed, test_frac=cfg["data"]["test_frac"],
                  stream_mode="uniform (Poissonised)", tau_per_feature=1.0, M_over_N_grid=[33, 100, 300, 900, 2700, 8100],
                  S_grid="powers of 2 from 8 to N, plus N", count_sketch_rows=3, jl_sparse_rows=4, hash_logreg_max_S={512: 16384, 2048: 4096},
                  synthetic=dict(pi=0.1, snr=2.5, n_train={512: 600, 2048: 160}, n_test={512: 300, 2048: 80}),
                  S_star_rule="smallest S with mean balanced accuracy (5 seeds) >= exact centroid - 0.01",
                  bytes_rule="8 bytes per float64 counter, 4 per int32 index, 8 per 64-bit hash seed")
    if args.quick:
        route2["M_over_N_grid"] = [100, 900]
    cfg["route2"] = route2

    t_all = time.time()
    X, genes, labels, summary = load_pbmc3k()
    by_K = []
    for K in route2["K_values"]:
        N = K * (K + 1) // 2
        S_grid = [2**e for e in range(3, int(math.floor(math.log2(N))) + 1)]
        if S_grid[-1] != N:
            S_grid.append(N)
        q = qledger(N, eps=cfg["resources"]["epsilon"], Q=cfg["resources"]["queries"], pi=cfg["resources"]["pi_rare"],
                    gate_rate_hz=cfg["resources"]["logical_gate_rate_hz"])
        blk = dict(K=K, N=N, S_grid=S_grid, quantum=dict(index_qubits=q["index_qubits"], logical_qubits=q["logical_qubits"],
                   samples_balanced=q["samples_balanced"], samples_rare=q["samples_rare"], classical_bytes_for_Delta=F64 * N),
                   real=[], synthetic=None)
        for ti, pos in enumerate(POSITIVES):
            t = time.time()
            d = build_real_task(X, genes, labels, pos, K, seed, cfg["data"]["test_frac"], cfg["data"]["min_cells_per_gene"])
            log(f"K={K} {pos}: pair features built in {time.time() - t:.0f} s (train nnz {d['P_train'].nnz:,}, test nnz {d['P_test'].nnz:,})")
            r = run_task(f"K={K} {pos}", d["P_train"], d["y_train"], d["P_test"], d["y_test"], d["pi_hat"], N, S_grid,
                         route2["M_over_N_grid"], n_seeds, seed * 1000 + K * 10 + ti, d["abstain_train"], d["abstain_test"],
                         do_logreg=not args.no_logreg, do_jl_dense=(K <= 512), quick=args.quick,
                         logreg_max_iter=200 if K > 512 else 300, hash_logreg_max_S=16384 if K <= 512 else 4096)
            r["positive"] = pos
            r["genes_head"] = d["genes"][:12]
            blk["real"].append(r)
            del d
            _dump(cfg, summary, by_K + [blk], partial=True)
        if not args.no_synthetic:
            sy = route2["synthetic"]
            d = build_synthetic(N, sy["n_train"].get(K, 300), sy["n_test"].get(K, 150), sy["pi"], sy["snr"], seed)
            r = run_task(f"K={K} synthetic", d["P_train"], d["y_train"], d["P_test"], d["y_test"], d["pi_hat"], N, S_grid,
                         route2["M_over_N_grid"], n_seeds, seed * 1000 + K * 10 + 9, do_logreg=False, do_jl_dense=(K <= 512),
                         quick=args.quick, P_thr=d["P_thr"], y_thr=d["y_thr"])
            r["params"] = d["params"]
            blk["synthetic"] = r
            del d
        by_K.append(blk)
        _dump(cfg, summary, by_K, partial=(K != route2["K_values"][-1]))
    log(f"all done in {(time.time() - t_all) / 60:.1f} min")


def _finalise(by_K):
    """Recompute every S* variant from the stored rows: vs the exact centroid (the user's rule), vs the S = N stream
    estimate at the same M (same records, the fair memory-only comparison), stable versions (all larger S also
    inside the band) and the AUPRC version. The test-set granularity is stored next to them."""
    for blk in by_K:
        tasks = list(blk["real"]) + ([blk["synthetic"]] if blk.get("synthetic") else [])
        for tk in tasks:
            ex_ba, ex_ap = tk["exact_centroid"]["balanced_accuracy"], tk["exact_centroid"]["auprc"]
            st_ba = tk["stream_full"]["balanced_accuracy"]
            cell = 0.5 / max(tk["n_test_pos"], 1)                      # one test positive in balanced accuracy
            tol_cell = max(0.01, cell)
            tk["granularity_balanced_accuracy_one_test_cell"] = cell
            tk["tolerance_cell"] = tol_cell
            for key, m in tk["methods"].items():
                rows = m["rows"]
                N = tk["N"]
                for name, target, k, fn, tol in [("S_star", ex_ba, "balanced_accuracy_mean", s_star, 0.01),
                                                 ("S_star_stable", ex_ba, "balanced_accuracy_mean", s_star_stable, 0.01),
                                                 ("S_star_stream", st_ba, "balanced_accuracy_mean", s_star, 0.01),
                                                 ("S_star_stream_stable", st_ba, "balanced_accuracy_mean", s_star_stable, 0.01),
                                                 ("S_star_auprc", ex_ap, "auprc_mean", s_star, 0.01),
                                                 ("S_star_cell", ex_ba, "balanced_accuracy_mean", s_star, tol_cell),
                                                 ("S_star_cell_stable", ex_ba, "balanced_accuracy_mean", s_star_stable, tol_cell)]:
                    S, B = fn(rows, target, key=k, tol=tol)
                    m[name] = S
                    m[name + "_over_N"] = (S / N) if S else None
                    if name == "S_star":
                        m["bytes_at_S_star"] = B
                    if name == "S_star_cell_stable":
                        m["bytes_at_S_star_cell_stable"] = B
    return by_K


def _summary_table(by_K):
    rows = []
    for blk in by_K:
        tasks = list(blk["real"]) + ([blk["synthetic"]] if blk.get("synthetic") else [])
        for tk in tasks:
            for key, m in tk["methods"].items():
                rows.append(dict(K=blk["K"], N=blk["N"], task=tk["task"], method=key, exact_balanced_accuracy=tk["exact_centroid"]["balanced_accuracy"],
                                 stream_full_balanced_accuracy=tk["stream_full"]["balanced_accuracy"], n_test_pos=tk["n_test_pos"],
                                 S_star=m["S_star"], S_star_over_N=m["S_star_over_N"], S_star_stable=m["S_star_stable"],
                                 S_star_stream=m["S_star_stream"], S_star_stream_stable=m["S_star_stream_stable"],
                                 S_star_stream_stable_over_N=m["S_star_stream_stable_over_N"], S_star_auprc=m["S_star_auprc"],
                                 tolerance_cell=tk["tolerance_cell"], S_star_cell=m["S_star_cell"], S_star_cell_stable=m["S_star_cell_stable"],
                                 S_star_cell_stable_over_N=m["S_star_cell_stable_over_N"],
                                 bytes_at_S_star=m["bytes_at_S_star"], quantum_index_qubits=blk["quantum"]["index_qubits"]))
    return rows


def _verdict(by_K):
    """Plain-language verdict computed from the numbers (used by the slide and the results note).
    Per class: the best one-pass method's S*/N under the granularity-aware rule (stable, vs the exact centroid,
    tolerance = max(0.01, one test positive)); hashing_logreg is not one-pass and is left out."""
    out = {}
    for blk in by_K:
        def best(tk):
            cands = []
            for k, m in tk["methods"].items():
                if k in ("hashing_logreg", "heavy_hitters_top", "jl_dense"):      # not one-pass / oracle bound / K=512 only
                    continue
                v = m.get("S_star_cell_stable_over_N")
                cands.append((v if v is not None else 2.0, k))
            v, k = min(cands)
            return dict(method=k, S_star_over_N=(v if v <= 1 else None), tolerance=tk["tolerance_cell"])
        real_best = {tk["positive"]: best(tk) for tk in blk["real"]}
        syn = None
        if blk.get("synthetic"):
            syn = {k: dict(S_star_over_N=m.get("S_star_over_N"), S_star_cell_stable_over_N=m.get("S_star_cell_stable_over_N"))
                   for k, m in blk["synthetic"]["methods"].items()}
        worst_real = max((v["S_star_over_N"] if v["S_star_over_N"] is not None else 1.0) for v in real_best.values())
        out[str(blk["K"])] = dict(best_method_per_class=real_best, worst_real_S_star_over_N=worst_real, synthetic=syn,
                                  memory_advantage_on_pbmc3k=("no" if worst_real <= 0.01 else "weak" if worst_real <= 0.1 else "possible"),
                                  rule="stable S* (all larger S inside the band) of the best one-pass method vs the exact centroid, "
                                       "tolerance = max(0.01, one test positive in balanced accuracy); "
                                       "a worst-case ratio <= 0.01 means every class is solved with <= 1 % of N counters")
    return out


def _dump(cfg, summary, by_K, partial=False):
    _finalise(by_K)
    body = dict(schema_version=1, experiment="e7_route2", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                partial=partial, config=cfg, dataset=summary, by_K=by_K, summary_table=_summary_table(by_K), verdict=_verdict(by_K),
                limits=["Louvain labels come from clustering the same expression data (circular)",
                        "rare-class test sets are tiny (DC = 11 cells, Mk = 5 cells): one cell = 0.1 (Mk) or 0.045 (DC) balanced accuracy",
                        "one split (config seed), 5 stream/hash seeds; a finite experiment is evidence, not proof",
                        "the stream is Poissonised (independent Poisson counts per entry), not drawn record by record",
                        "the quantum column is the classically simulated sketch at this N; nothing was run on hardware at this N",
                        "S counts counters; a qubit and a float are not the same unit"])
    out = ROOT / "results" / RESULT_NAME
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(_clean(body), separators=(",", ":")))
    log(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} kB){' [partial]' if partial else ''}")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ===================================================================================================================
# figure (JSON -> PNG; numpy + matplotlib only, so it also runs on a machine without the data)
# ===================================================================================================================
def figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads((ROOT / "results" / RESULT_NAME).read_text())
    assert d["schema_version"] == 1
    INK, GOLD, DIM, RED = "#08090B", "#D9A800", "#8A8984", "#D94F3D"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    style = {"count_sketch": (INK, "o-", "count-sketch (3 rows, median)"), "hashing": ("#4A4A48", "s-", "feature hashing (1 row)"),
             "jl_sparse": ("#8A8984", "^-", "sparse JL (4 rows)"), "jl_dense": ("#8A8984", "v--", "dense Gaussian JL"),
             "heavy_hitters_mg": (GOLD, "D-", "heavy hitters (Misra–Gries)"), "heavy_hitters_top": (GOLD, "d:", "heavy hitters (oracle top-S)"),
             "reservoir": (RED, "x-", "reservoir (S records)"), "hashing_logreg": ("#2F6DB5", "p--", "hashing + logreg (not one-pass)")}
    blocks = [b for b in d["by_K"] if (args.K is None or b["K"] == args.K)]
    ncol = max(len(b["real"]) + (1 if b.get("synthetic") else 0) for b in blocks)
    fig, axes = plt.subplots(len(blocks), ncol, figsize=(4.4 * ncol, 4.1 * len(blocks) + 0.6), sharey=True, squeeze=False)
    for row, blk in zip(axes, blocks):
        panels = [(t["positive"], t) for t in blk["real"]] + ([("synthetic control (no structure)", blk["synthetic"])] if blk.get("synthetic") else [])
        q = blk["quantum"]["index_qubits"]
        for ax, (title, tk) in zip(row, panels):
            N = tk["N"]
            ex = tk["exact_centroid"]["balanced_accuracy"]
            for key, m in tk["methods"].items():
                if key not in style:
                    continue
                col, fmt, lab = style[key]
                S = [r["S"] for r in m["rows"]]
                y = [r["balanced_accuracy_mean"] for r in m["rows"]]
                e = [r["balanced_accuracy_std"] for r in m["rows"]]
                ax.errorbar(S, y, yerr=e, fmt=fmt, color=col, ms=3.5, lw=1.3, capsize=0, label=lab, alpha=0.9 if "top" not in key else 0.7)
            ax.axhline(ex, color=INK, ls=":", lw=1)
            tol = tk.get("tolerance_cell", 0.01)
            if tol > 0.01:
                ax.axhspan(ex - tol, ex + tol, color=INK, alpha=0.035, lw=0)
            ax.axhspan(ex - 0.01, ex + 0.01, color=INK, alpha=0.07, lw=0)
            ax.axvline(N, color=INK, ls="--", lw=0.8)
            ax.text(N / 1.15, 0.515, f"S = N = {N:,}\n(store every feature)", ha="right", va="bottom", fontsize=7, color=INK)
            ax.axvline(q, color=RED, ls="--", lw=1.2)
            ax.text(q * 1.15, 0.515, f"quantum sketch:\n{q} index qubits", ha="left", va="bottom", fontsize=7, color=RED)
            ax.set_xscale("log", base=2)
            ax.set_ylim(0.48, 1.03)
            ax.set_xlabel("classical memory S (counters, log scale)")
            sub = f"K = {blk['K']} · N = {N:,} · π̂ = {tk['pi_hat']:.3f} · test positives {tk['n_test_pos']} · M = {tk['M_run_over_N']}·N"
            ax.set_title(f"{title}\n{sub}", fontsize=8.5, loc="left")
        row[0].set_ylabel("balanced accuracy (test, 5 seeds)")
        for ax in row[len(panels):]:
            ax.axis("off")
    seen, H, L = set(), [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); H.append(h); L.append(l)
    fig.legend(H, L, loc="lower center", ncol=min(4, len(L)), fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("E7 · how much classical memory does the gene-pair rare-cell task need?   dotted line: exact centroid (M→∞ sketch); "
                 "bands: ± 0.01 (dark) and ± one test positive (light) · S* = first S inside the band", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out = ROOT / "figures" / FIG_NAME
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    log(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--K", nargs="+", default=[512, 2048])
    r.add_argument("--seeds", default=5)
    r.add_argument("--quick", action="store_true")
    r.add_argument("--no-logreg", action="store_true")
    r.add_argument("--no-synthetic", action="store_true")
    f = sub.add_parser("figure")
    f.add_argument("--K", type=int, default=None)
    m = sub.add_parser("merge", help="merge the by_K blocks of several result files (one per K) into results/e7_route2.json")
    m.add_argument("files", nargs="+")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "figure":
        figure(a)
    else:
        parts = [json.loads(Path(f).read_text()) for f in a.files]
        by_K = sorted([b for p in parts for b in p["by_K"]], key=lambda b: b["K"])
        cfg = parts[-1]["config"]
        cfg["route2"]["K_values"] = [b["K"] for b in by_K]
        cfg["route2"]["merged_from"] = [str(f) for f in a.files]
        _dump(cfg, parts[-1]["dataset"], by_K, partial=any(p.get("partial") for p in parts))
