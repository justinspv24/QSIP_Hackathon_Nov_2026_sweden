"""E14 core: shared data, pairs, folds, nested-CV subsets, exact weighted ridge engine, memory ledgers.

Reused from E13 Stage 4 (imported, unchanged): load() (scvelo PBMC68k, log1p, genes with >= 1 count), paper_pairs()
(the paper's 100 draws, rng 42), the 46 unique pairs + multiplicities, sklearn ridge_cv (the gate), the paper's
sketch_utils (signed balanced hashing = 'jl' mode, the paper's seeds).

Folds: outer = StratifiedKFold(5), no shuffle (= cross_val_score(cv=5), as E13 / the paper). Every pair row gets an
outer block id b in 0..4. Nested CV: for outer fold k (test = block k) the inner validation folds are the other
blocks j != k (inner train = blocks not in {k, j}). Subset list (k, j): j = -1 means "full outer training fold,
evaluated on the outer test block". Hyperparameters are chosen ONLY from inner-validation accuracy; the test block is
never touched by tuning.

Ridge engine: exact solution of  min_w,b  sum_i sw_i (x_i.w + b - y_i)^2 + alpha |w|^2  with y in {-1,+1} and
sw = balanced class weights n/(2 n_c) of the training subset = sklearn RidgeClassifier(class_weight='balanced').
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
E14 = HERE.parent
ROOT = E14.parent
sys.path.insert(0, str(ROOT / "HPC_E13" / "scripts"))
import e13_stage4 as S4  # noqa: E402  (reused unchanged)
import e13_core as C13  # noqa: E402
import sketch_utils  # noqa: E402  (paper code, via e13_stage4's sys.path)

# data: reuse the E13 worktree download (sha256 checked against e13_stage4.json)
DATA = ROOT.parents[1] / "qos_cells_e13" / "qos_cells" / "HPC_E13" / "data" / "scvelo" / "pbmc68k_scvelo.h5ad"
S4.DATA = DATA
SHA256 = "c93b1ccad909b6a41539a57975737bd946ea1becce066c250aca129d8dfa26fb"
E13_JSON = ROOT / "HPC_E13" / "results" / "e13_stage4.json"
S_GRID = [16, 32, 44, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
FULL_RIDGE = 0.90146   # E13 Stage 4, weighted over the 46 pairs


def log(msg):
    print(f"[e14 {dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_ram_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def check_sha():
    import hashlib
    h = hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert h == SHA256, f"data sha256 mismatch {h}"
    return h


def load_all():
    X, y, names, cnt = S4.load()
    pairs = S4.paper_pairs(len(names))
    upairs = sorted(set(pairs))
    mult = [pairs.count(p) for p in upairs]
    return X, y, names, upairs, mult


def pair_data(X, y, upairs, pi):
    c1, c2 = upairs[pi]
    m = (y == c1) | (y == c2)
    Xp = X[m].tocsr().astype(np.float32)
    yp = (y[m] == c2).astype(np.int64)
    return Xp, yp


def blocks(yp):
    from sklearn.model_selection import StratifiedKFold
    b = np.empty(len(yp), dtype=np.int64)
    for k, (_, te) in enumerate(StratifiedKFold(5).split(np.zeros(len(yp)), yp)):
        b[te] = k
    return b


def subsets(n_inner=4):
    """list of (k, j): outer fold k, inner validation block j (-1 = full outer train, eval on test block k).
    n_inner < 4 uses the next n_inner blocks (k+1, k+2, ...) mod 5 as inner folds."""
    out = []
    for k in range(5):
        out.append((k, -1))
        for d in range(1, n_inner + 1):
            out.append((k, (k + d) % 5))
    return out


def subset_masks(b, subs):
    tr = np.zeros((len(subs), len(b)), dtype=bool)
    ev = np.zeros((len(subs), len(b)), dtype=bool)
    for u, (k, j) in enumerate(subs):
        if j < 0:
            tr[u] = b != k
            ev[u] = b == k
        else:
            tr[u] = (b != k) & (b != j)
            ev[u] = b == j
    return tr, ev


def hashed(Xp, S, seed):
    """the paper's signed balanced hashing (sketch_utils 'jl', sparsity 1) of the pair matrix, CSR float32."""
    Xs, _ = sketch_utils.sketch_features(Xp, S, "jl", seed=seed)
    Xs = sp.csr_matrix(Xs, dtype=np.float32)
    Xs.sum_duplicates()
    Xs.eliminate_zeros()
    return Xs


def acc_bacc(yt, pred):
    yt = np.asarray(yt)
    pred = np.asarray(pred)
    a = float((yt == pred).mean())
    rec = [float((pred[yt == c] == c).mean()) for c in (0, 1) if (yt == c).any()]
    return a, float(np.mean(rec))


# ---------------------------------------------------------------------------------------------------------------
# exact weighted ridge (GPU via cupy if xp is cupy) on a dense feature matrix; nested-k prefix solves
# ---------------------------------------------------------------------------------------------------------------
def ridge_prefix(xp, F, y, tr, ev, k_list, alphas=ALPHAS):
    """F: (n, kmax) float32 (xp array), columns in ranking order. y: (n,) 0/1 xp int. tr/ev: boolean xp masks.
    Returns preds dict {(k, alpha): predicted 0/1 on ev rows (xp)}  computed with the leading k columns.
    Weighted, centred Gram in float32 (centred first), solve in float64 by Cholesky."""
    Ft = F[tr]
    yt = y[tr]
    n = Ft.shape[0]
    n1 = int(yt.sum())
    n0 = n - n1
    sw = xp.where(yt == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(xp.float32)
    ys = (2 * yt - 1).astype(xp.float32)
    xbar = (sw[:, None] * Ft).sum(0) / sw.sum()            # weighted mean
    ybar = float((sw * ys).sum() / sw.sum())                 # = 0 for balanced weights
    A = (Ft - xbar[None, :]) * xp.sqrt(sw)[:, None]
    Gm = (A.T @ A).astype(xp.float64)
    r = (A.T @ (xp.sqrt(sw) * (ys - ybar))).astype(xp.float64)
    Fe = F[ev]
    out = {}
    for k in k_list:
        G = Gm[:k, :k]
        I = xp.eye(k, dtype=xp.float64)
        for al in alphas:
            L = xp.linalg.cholesky(G + al * I)
            w = xp.linalg.solve(L.T, xp.linalg.solve(L, r[:k]))
            b = ybar - float(xbar[:k].astype(xp.float64) @ w)
            s = Fe[:, :k].astype(xp.float64) @ w + b
            out[(k, al)] = (s > 0).astype(xp.int64)
    return out


def ridge_prefix_fast(xp, F, y, tr, ev, k_list, alphas=ALPHAS):
    """Same solutions as ridge_prefix, all k at once. Cholesky of the leading k x k block of (G + aI) is the leading
    block of the full factor L, so with z = L^-1 r, M = L^-1 Fe^T, m = L^-1 xbar:
      score_k(row) = sum_{j<k} M[j,row] z[j] + ybar - sum_{j<k} m[j] z[j].
    Returns (acc, bacc) arrays of shape (len(alphas), len(k_list)) as numpy."""
    from cupyx.scipy.linalg import solve_triangular as st_gpu
    Ft = F[tr]
    yt = y[tr]
    n = Ft.shape[0]
    n1 = int(yt.sum())
    n0 = n - n1
    sw = xp.where(yt == 1, n / (2.0 * n1), n / (2.0 * n0)).astype(xp.float32)
    ys = (2 * yt - 1).astype(xp.float32)
    xbar = (sw[:, None] * Ft).sum(0) / sw.sum()
    ybar = float((sw * ys).sum() / sw.sum())
    A = (Ft - xbar[None, :]) * xp.sqrt(sw)[:, None]
    Gm = (A.T @ A).astype(xp.float64)
    r = (A.T @ (xp.sqrt(sw) * (ys - ybar))).astype(xp.float64)
    Fe = F[ev].astype(xp.float64)
    ye = y[ev]
    kmax = F.shape[1]
    kk = xp.asarray(np.array(k_list) - 1)
    I = xp.eye(kmax, dtype=xp.float64)
    xb = xbar.astype(xp.float64)
    accs, baccs = [], []
    for al in alphas:
        L = xp.linalg.cholesky(Gm + al * I)
        R = st_gpu(L, xp.ascontiguousarray(xp.concatenate([r[:, None], xb[:, None], Fe.T], axis=1)), lower=True)
        z, m, M = R[:, 0], R[:, 1], R[:, 2:]
        S = xp.cumsum(M * z[:, None], axis=0)[kk]            # (nk, n_e)
        bb = ybar - xp.cumsum(m * z)[kk]
        pred = ((S + bb[:, None]) > 0).astype(xp.int64)
        cor = pred == ye[None, :]
        accs.append(cor.mean(1))
        pos = ye == 1
        rc = [cor[:, pos].mean(1) if bool(pos.any()) else None, cor[:, ~pos].mean(1) if bool((~pos).any()) else None]
        rc = [v for v in rc if v is not None]
        baccs.append(sum(rc) / len(rc))
    return xp.asnumpy(xp.stack(accs)) if xp is not np else np.stack(accs), \
        xp.asnumpy(xp.stack(baccs)) if xp is not np else np.stack(baccs)


def git_commit():
    return C13.git_commit()


def hardware():
    hw = C13.hardware()
    try:
        import subprocess
        hw["gpus"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().split("\n")
    except Exception:
        pass
    return hw


def clean(o):
    return C13.clean(o)


def write_json(path, body):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(body), separators=(",", ":")))
    os.replace(tmp, path)


def weighted_mean_sem(vals_by_pair, mult):
    """vals_by_pair: {pair index: scalar}; weights = paper draw multiplicities. Same formula as E13 merge()."""
    idx = sorted(vals_by_pair)
    if not idx:
        return None
    v = np.array([vals_by_pair[i] for i in idx])
    w = np.array([mult[i] for i in idx], dtype=float)
    m = float(np.sum(w * v) / w.sum())
    sem = float(np.sqrt(np.sum(w * (v - m) ** 2) / w.sum()) / math.sqrt(len(idx)))
    return dict(mean=m, sem=sem, unweighted_mean=float(v.mean()), n_pairs=len(idx))
