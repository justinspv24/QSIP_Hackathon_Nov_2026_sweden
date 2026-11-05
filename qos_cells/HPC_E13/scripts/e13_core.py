"""E13 core: one (dataset, feature set, task, split) job = E7's route-2 memory test, on PBMC68k or PBMC3k.

Reused unchanged from E7 (scripts/09_route2_classical_memory.py): pair_features_csr, the sketches (count-sketch 3 rows,
feature hashing, sparse JL 4 rows, Misra-Gries S/2+S/2, oracle top-S), hashing+logreg, class-balanced logreg,
evaluate_columns (threshold on train, balanced accuracy + AUPRC on test), summarise, s_star / s_star_stable, the
byte ledger. Reused from qos_cells: select_genes (train-only |mu1-mu0| on amplitude features), stratified_split,
class_weights.

One change, for speed only: the Poissonised uniform stream. E7 draws an independent Poisson(lam) count for every
nonzero (cell, feature) entry (lam = M/(N n)). Here the same law is sampled as R ~ Poisson(lam * nnz) records, each
placed on a uniformly random nonzero entry. Poisson splitting makes the per-entry counts independent Poisson(lam):
the two samplers have the same distribution (checked in e13_selftest.py). Cost O(R) instead of O(nnz) per draw.

Feature sets (amplitude-normalised per cell, as in E7):
  genes   : p_g = x_g^2 / sum x^2 over the top-K genes (K = 1024 -> N = 1024)
  pairs   : p_(g,h) = x_g^2 x_h^2 / sum_(g'<=h') x_g'^2 x_h'^2 over g <= h of the top-K genes (N = K(K+1)/2)
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

E13 = Path(__file__).resolve().parents[1]           # qos_cells/HPC_E13
ROOT = E13.parent                                    # qos_cells
sys.path.insert(0, str(ROOT))
from qos_cells import load_config  # noqa: E402
from qos_cells.load_data import select_genes, stratified_split  # noqa: E402
from qos_cells.stream import class_weights  # noqa: E402

_spec = importlib.util.spec_from_file_location("e7", ROOT / "scripts" / "09_route2_classical_memory.py")
E7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E7)

DATA68K = E13 / "data" / "pbmc68k.h5ad"
TASKS = {"pbmc68k": ["CD4+ T Helper2", "CD34+", "CD4+/CD45RA+/CD25- Naive T", "CD19+ B"],
         "pbmc3k": ["Megakaryocytes", "Dendritic cells", "B cells"]}
M_GRID = [100, 300, 900, 2700, 8100, 24300, 72900, 218700]
METHODS = [("count_sketch", "count-sketch (3 rows, median)"), ("hashing", "feature hashing (signed, 1 row)"),
           ("jl_sparse", "sparse JL projection (4 rows)"), ("heavy_hitters_mg", "heavy hitters: Misra-Gries (S/2 + S/2)"),
           ("heavy_hitters_top", "heavy hitters: oracle top-S |Delta|"), ("hashing_logreg", "feature hashing + logistic regression")]


def log(msg):
    print(f"[e13 {dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# run metadata
# ---------------------------------------------------------------------------------------------------------------
def git_commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "."], cwd=ROOT) != 0
        return h + ("-dirty" if dirty else "")
    except Exception as e:
        return f"unknown ({e})"


def hardware():
    hw = dict(hostname=socket.gethostname(), cpu_count=os.cpu_count(), platform=platform.platform(), python=sys.version.split()[0])
    try:
        hw["cpu_model"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
        hw["ram_gb"] = round(int([l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal")][0]) / 2**20, 1)
        hw["cpu_affinity"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    return hw


def peak_ram_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ---------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------
def load_dataset(name):
    """-> X (csr float32, log1p of counts scaled to 1e4 per cell), genes, labels, summary."""
    if name == "pbmc3k":
        from qos_cells.load_data import load_pbmc3k
        X, genes, labels, summary = load_pbmc3k(ROOT / "data" / "pbmc3k_processed.h5ad")
        return X, genes, labels, summary
    import anndata as ad
    a = ad.read_h5ad(DATA68K)
    X = sp.csr_matrix(a.X, dtype=np.float32)
    tot = np.asarray(X.sum(1)).ravel()
    X = sp.diags((1e4 / np.maximum(tot, 1)).astype(np.float32)) @ X
    X.data = np.log1p(X.data)
    X = sp.csr_matrix(X, dtype=np.float32)
    labels = np.asarray(a.obs["celltype"].astype(str))
    info = json.loads((E13 / "results" / "e13_data.json").read_text())
    summary = dict(dataset="pbmc68k", n_cells=int(X.shape[0]), n_genes=int(X.shape[1]), nnz=int(X.nnz),
                   sha256_matrix_tgz=info["sources"]["matrix"]["sha256"], sha256_annotation=info["sources"]["annotation"]["sha256"],
                   class_counts=info["class_counts"])
    return X, np.asarray(a.var_names, dtype=str), labels, summary


def gene_features_csr(Xsel):
    """p_g = x_g^2 / sum x^2 over the selected genes, as csr (float32)."""
    Xs = sp.csr_matrix(Xsel, dtype=np.float64)
    sq = Xs.multiply(Xs).tocsr()
    n2 = np.asarray(sq.sum(1)).ravel()
    abstain = n2 == 0
    n2[abstain] = 1.0
    P = sp.csr_matrix(sp.diags(1.0 / n2) @ sq, dtype=np.float32)
    P.sort_indices()
    return P, abstain


def pair_features_fast(Xsel):
    """Numba version of E7.pair_features_csr (same indices, same float32 values up to rounding; checked in e13_selftest_pairs.py)."""
    import numba

    Xsel = sp.csr_matrix(Xsel)
    Xsel.sort_indices()
    n, K = Xsel.shape
    N = K * (K + 1) // 2
    m = np.diff(Xsel.indptr).astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(m * (m + 1) // 2)
    indices = np.empty(indptr[-1], dtype=np.int32)
    data = np.empty(indptr[-1], dtype=np.float32)
    _pairs_kernel(Xsel.indptr.astype(np.int64), Xsel.indices.astype(np.int64), Xsel.data.astype(np.float64), indptr, K, indices, data)
    P = sp.csr_matrix((data, indices, indptr), shape=(n, N))
    P.has_sorted_indices = True
    return P, m == 0


def _make_kernel():
    import numba

    @numba.njit(parallel=True, cache=True)
    def kern(xp, xi, xd, op, K, oi, od):
        n = len(xp) - 1
        for i in numba.prange(n):
            a0, a1 = xp[i], xp[i + 1]
            mi = a1 - a0
            if mi == 0:
                continue
            tot = 0.0
            for a in range(mi):
                sa = xd[a0 + a] ** 2
                for b in range(a, mi):
                    tot += sa * xd[a0 + b] ** 2
            o = op[i]
            for a in range(mi):
                g = xi[a0 + a]
                sa = xd[a0 + a] ** 2
                for b in range(a, mi):
                    h = xi[a0 + b]
                    oi[o] = g * K - g * (g - 1) // 2 + (h - g)
                    od[o] = sa * xd[a0 + b] ** 2 / tot
                    o += 1
    return kern


_KERN = None


def _pairs_kernel(*args):
    global _KERN
    if _KERN is None:
        _KERN = _make_kernel()
    _KERN(*args)


def split_indices(y, split, test_frac, base_seed):
    """split 0 = the config seed (E7's split); split k > 0 = seed base_seed + 1000 k."""
    return stratified_split(y, test_frac, base_seed if split == 0 else base_seed + 1000 * split)


def build(X, genes, labels, positive, feat, K, split, test_frac, min_cells, base_seed):
    y = (labels == positive).astype(int)
    tr, te = split_indices(y, split, test_frac, base_seed)
    g = select_genes(X[tr], y[tr], K, min_cells)                      # TRAIN ONLY
    f = gene_features_csr if feat == "genes" else pair_features_fast
    P_tr, ab_tr = f(X[tr][:, g])
    P_te, ab_te = f(X[te][:, g])
    return dict(P_train=P_tr, y_train=y[tr], P_test=P_te, y_test=y[te], pi_hat=float(y[tr].mean()),
                abstain_train=ab_tr, abstain_test=ab_te, genes=[str(s) for s in genes[g]])


# ---------------------------------------------------------------------------------------------------------------
# the stream (compound-Poisson sampler, same law as E7.poisson_stream)
# ---------------------------------------------------------------------------------------------------------------
class Stream:
    def __init__(self, P, w_cell):
        P = sp.csr_matrix(P)
        self.N, self.n, self.nnz = P.shape[1], P.shape[0], P.nnz
        row = np.repeat(np.arange(P.shape[0], dtype=np.int32), np.diff(P.indptr))
        self.col = P.indices.astype(np.int32)
        self.val = (w_cell[row] * P.data.astype(np.float64))         # kick per record on this entry (before N/M)
        self.pos = w_cell[row] > 0

    def draw(self, M, rng, chunk=200_000_000):
        lam = M / (self.N * self.n)
        R = int(rng.poisson(lam * self.nnz))
        phi_pos, phi_neg = np.zeros(self.N), np.zeros(self.N)
        for a in range(0, R, chunk):
            k = rng.integers(0, self.nnz, min(chunk, R - a))
            v, c, p = self.val[k], self.col[k], self.pos[k]
            phi_pos += np.bincount(c[p], weights=v[p], minlength=self.N)
            phi_neg += np.bincount(c[~p], weights=-v[~p], minlength=self.N)
        s = self.N / M
        return s * (phi_pos - phi_neg), s * phi_pos, s * phi_neg, R


def _make_spmm():
    import numba

    @numba.njit(parallel=True, cache=True)
    def spmm(indptr, indices, data, D, out):
        n = len(indptr) - 1
        k = D.shape[1]
        for i in numba.prange(n):
            for a in range(indptr[i], indptr[i + 1]):
                j = indices[a]
                v = data[a]
                for c in range(k):
                    out[i, c] += v * D[j, c]
    return spmm


_SPMM = None


def spmm(P, D):
    """P (csr) @ D (dense N x k), multi-threaded (numba), float64 accumulate."""
    global _SPMM
    if _SPMM is None:
        _SPMM = _make_spmm()
    D = np.ascontiguousarray(D, dtype=np.float64)
    out = np.zeros((P.shape[0], D.shape[1]))
    _SPMM(P.indptr, P.indices, P.data, D, out)
    return out


def evaluate_columns(P_thr, y_thr, P_test, y_test, D, abstain_thr=None, abstain_test=None):
    """Same as E7.evaluate_columns (threshold on the train scores, BA + AUPRC on test) with a threaded SpMM."""
    from qos_cells.readout import best_threshold, evaluate

    D = np.asarray(D, dtype=np.float64)
    out = []
    for a in range(0, D.shape[1], 16):
        s_tr, s_te = spmm(P_thr, D[:, a:a + 16]), spmm(P_test, D[:, a:a + 16])
        for c in range(s_tr.shape[1]):
            x, z = s_tr[:, c].copy(), s_te[:, c].copy()
            if abstain_thr is not None:
                x[abstain_thr] = -np.inf
            if abstain_test is not None:
                z[abstain_test] = -np.inf
            ev = evaluate(z, y_test, best_threshold(x, y_thr))
            out.append(dict(balanced_accuracy=ev["balanced_accuracy"], auprc=ev["auprc"]))
    return out


def _make_hash_kernel():
    import numba

    @numba.njit(parallel=True, cache=True)
    def k(indptr, indices, data, hb, sg, out):
        for i in numba.prange(len(indptr) - 1):
            for a in range(indptr[i], indptr[i + 1]):
                j = indices[a]
                out[i, hb[j]] += sg[j] * data[a]
    return k


_HK = None


def hashed_max(P, seeds, Smax):
    """Signed feature hashing of every row of P into Smax buckets (E7 hbucket/hsign with seeds[0], seeds[1])."""
    global _HK
    if _HK is None:
        _HK = _make_hash_kernel()
    j = np.arange(P.shape[1])
    hb, sg = E7.hbucket(j, seeds[0], Smax), E7.hsign(j, seeds[1])
    out = np.zeros((P.shape[0], Smax), dtype=np.float64)
    _HK(P.indptr, P.indices, P.data, hb, sg, out)
    return out


def fold(H, S):
    """Features at S buckets from those at Smax (powers of 2): mix(j) % S == (mix(j) % Smax) % S."""
    n, Smax = H.shape
    return H.reshape(n, Smax // S, S).sum(1).astype(np.float32) if S < Smax else H.astype(np.float32)


def d_eff(D):
    D = np.asarray(D, dtype=np.float64)
    return float(np.sum(D**2) ** 2 / np.sum(D**4))


# ---------------------------------------------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------------------------------------------
def run_task(name, d, N, S_grid, n_seeds, base_seed, hash_logreg_max_S, logreg_max_iter, do_logreg=True, m_cap_records=3e9,
             do_logreg_full=True):
    from qos_cells.sketch import rescale_safe

    t0 = time.time()
    P_train, y_train, P_test, y_test = d["P_train"], d["y_train"], d["P_test"], d["y_test"]
    ab_tr, ab_te, pi_hat = d["abstain_train"], d["abstain_test"], d["pi_hat"]
    n_train = P_train.shape[0]
    w_cell = class_weights(pi_hat)[y_train]
    res = dict(task=name, N=int(N), index_qubits=int(math.ceil(math.log2(N))), logical_qubits_approx=2 * int(math.ceil(math.log2(N))),
               n_train=int(n_train), n_test=int(P_test.shape[0]), n_train_pos=int(y_train.sum()), n_test_pos=int(y_test.sum()),
               pi_hat=pi_hat, base_rate_test=float(y_test.mean()), abstain_rate_test=float(ab_te.mean()),
               abstain_rate_test_pos=float(ab_te[y_test == 1].mean()), train_density=float(P_train.nnz / (n_train * N)),
               nnz_train=int(P_train.nnz), nnz_test=int(P_test.nnz))
    ind1 = (y_train == 1).astype(np.float32) / max(int(y_train.sum()), 1)
    ind0 = (y_train == 0).astype(np.float32) / max(int((y_train == 0).sum()), 1)
    delta = np.asarray(P_train.T @ ind1, dtype=np.float64) - np.asarray(P_train.T @ ind0, dtype=np.float64)
    ev = evaluate_columns(P_train, y_train, P_test, y_test, delta[:, None], ab_tr, ab_te)[0]
    res["exact_centroid"] = dict(**ev, **E7.ledger(floats=N))
    res["d_eff"] = d_eff(delta)
    res["d_eff_over_N"] = res["d_eff"] / N
    exact_ba = ev["balanced_accuracy"]
    log(f"{name}: exact centroid BA {exact_ba:.4f} AUPRC {ev['auprc']:.4f}  d_eff {res['d_eff']:.1f} (N={N:,} n_tr={n_train} pos={int(y_train.sum())})")
    if do_logreg and do_logreg_full:
        t = time.time()
        scale = 1.0 / float(P_train.data.mean())
        res["logreg_full"] = dict(**E7.logreg_balanced(P_train, y_train, P_test, y_test, scale, max_iter=logreg_max_iter),
                                  **E7.ledger(floats=N + 1), feature_scale=scale, seconds=round(time.time() - t, 1))
        log(f"{name}: logreg all N BA {res['logreg_full']['balanced_accuracy']:.4f} AUPRC {res['logreg_full']['auprc']:.4f} "
            f"({res['logreg_full']['seconds']} s, it {res['logreg_full']['n_iter']})")

    # ---- M selection (E7 rule + E10's rel. error < 1; stop one grid step after convergence) ------------------------
    stream = Stream(P_train, w_cell)
    rngs = [np.random.default_rng(np.random.SeedSequence([base_seed, 7, s])) for s in range(n_seeds)]
    msel, phis, conv_i = [], {}, None
    for mi, mo in enumerate(M_GRID):
        M = int(mo * N)
        if M / (N * n_train) * P_train.nnz > m_cap_records:
            log(f"{name}: M/N={mo} would need > {m_cap_records:.0e} records per seed; stop")
            break
        cols, extra, recs = [], [], []
        for s in range(n_seeds):
            phi, ppos, pneg, R = stream.draw(M, rngs[s])
            phis[(mo, s)] = (phi, ppos)
            cols.append(phi)
            recs.append(R)
            extra.append(np.sin(rescale_safe(phi)[0]))
        evs = evaluate_columns(P_train, y_train, P_test, y_test, np.stack(cols, 1), ab_tr, ab_te)
        q = evaluate_columns(P_train, y_train, P_test, y_test, np.stack(extra, 1), ab_tr, ab_te)
        row = dict(M_over_N=mo, M=M, nonzero_records_mean=float(np.mean(recs)),
                   balanced_accuracy_mean=float(np.mean([e["balanced_accuracy"] for e in evs])),
                   balanced_accuracy_std=float(np.std([e["balanced_accuracy"] for e in evs])),
                   auprc_mean=float(np.mean([e["auprc"] for e in evs])),
                   balanced_accuracy_seeds=[round(e["balanced_accuracy"], 4) for e in evs],
                   rel_error_mean=float(np.mean([np.linalg.norm(c - delta) / np.linalg.norm(delta) for c in cols])),
                   quantum_sin_readout_balanced_accuracy_mean=float(np.mean([e["balanced_accuracy"] for e in q])),
                   quantum_sin_readout_auprc_mean=float(np.mean([e["auprc"] for e in q])))
        msel.append(row)
        log(f"{name}: M/N={mo:>6} stream BA {row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f} "
            f"rel.err {row['rel_error_mean']:.3f} ({row['nonzero_records_mean']:.2e} records)")
        for k in [k for k in phis if k[0] != mo and (conv_i is None or k[0] != M_GRID[conv_i])]:
            del phis[k]
        if conv_i is None and row["balanced_accuracy_mean"] >= exact_ba - 0.01 and row["rel_error_mean"] < 1:
            conv_i = mi
        elif conv_i is not None:
            break                                                      # one grid step beyond the first converged M
    res["M_selection"] = msel
    res["M_rule"] = "first M/N with stream BA >= exact - 0.01 and rel. error < 1; run at one grid step beyond (or the last M tried)"
    res["M_converged_over_N"] = M_GRID[conv_i] if conv_i is not None else None
    run_row = msel[-1]
    mo_run = run_row["M_over_N"]
    res.update(M_run=run_row["M"], M_run_over_N=mo_run)
    res["stream_full"] = dict(balanced_accuracy=run_row["balanced_accuracy_mean"], auprc=run_row["auprc_mean"],
                              balanced_accuracy_std=run_row["balanced_accuracy_std"], **E7.ledger(floats=N),
                              nonzero_records_mean=run_row["nonzero_records_mean"], rel_error_mean=run_row["rel_error_mean"])
    res["quantum_sim"] = dict(index_qubits=res["index_qubits"], logical_qubits_approx=res["logical_qubits_approx"],
                              balanced_accuracy=run_row["quantum_sin_readout_balanced_accuracy_mean"],
                              auprc=run_row["quantum_sin_readout_auprc_mean"], M=run_row["M"],
                              note="classical simulation of the oracle sketch (phases phi, readout p.sin(c phi)); not run on hardware")
    phis = {s: phis[(mo_run, s)] for s in range(n_seeds)}

    # ---- memory-limited methods ------------------------------------------------------------------------------
    methods = {}
    mu1 = np.asarray(P_train.T @ ind1, dtype=np.float64)
    mu0 = np.asarray(P_train.T @ ind0, dtype=np.float64)
    for mi_, (key, label) in enumerate(METHODS):
        if key == "hashing_logreg" and not do_logreg:
            continue
        t = time.time()
        grid = [S for S in S_grid if key != "hashing_logreg" or S <= hash_logreg_max_S]
        out = {}
        # source "stream": the sketch of the M_run stream (E7 protocol);  "Minf": the sketch of the exact Delta
        # (M -> infinity; isolates memory from sampling noise). hashing_logreg is trained on the cells: one source.
        for src in (["stream"] if key == "hashing_logreg" else ["stream", "Minf"]):
            per_seed = []
            for s in range(n_seeds):
                if src == "stream":
                    phi, ppos = phis[s]
                    pneg = ppos - phi
                else:
                    phi, ppos, pneg = delta, mu1, mu0
                rng = np.random.default_rng(np.random.SeedSequence([base_seed, 11, s, mi_]))
                seeds = rng.integers(0, 2**62, size=16)
                metrics = {}
                if key == "hashing_logreg":
                    scale = 1.0 / float(P_train.data.mean())
                    Htr_max, Hte_max = hashed_max(P_train, seeds, max(grid)), hashed_max(P_test, seeds, max(grid))
                    for S in grid:
                        Htr, Hte = fold(Htr_max, S), fold(Hte_max, S)       # == E7's hbucket(j, seed, S) features
                        ev_ = E7.logreg_balanced(Htr, y_train, Hte, y_test, scale, max_iter=logreg_max_iter)
                        metrics[S] = dict(balanced_accuracy=ev_["balanced_accuracy"], auprc=ev_["auprc"], mem=E7.ledger(floats=S + 1, seeds=2))
                    per_seed.append(metrics)
                    continue
                cols, mems = [], []
                for S in grid:
                    if key == "count_sketch":
                        dd, m = E7.sk_count_sketch(phi, S, seeds)
                    elif key == "hashing":
                        dd, m = E7.sk_hashing(phi, S, seeds)
                    elif key == "jl_sparse":
                        dd, m = E7.sk_jl_sparse(phi, S, seeds)
                    elif key == "heavy_hitters_mg":
                        dd, m = E7.sk_heavy_hitters_mg(ppos, pneg, S)
                    else:
                        dd, m = E7.sk_heavy_hitters_top(phi, S)
                    cols.append(dd)
                    mems.append(m)
                evs = evaluate_columns(P_train, y_train, P_test, y_test, np.stack(cols, 1), ab_tr, ab_te)
                for S, m, e in zip(grid, mems, evs):
                    metrics[S] = dict(balanced_accuracy=e["balanced_accuracy"], auprc=e["auprc"], mem=m)
                per_seed.append(metrics)
            out[src] = E7.summarise(per_seed)
        methods[key] = dict(label=label, rows=out["stream"], seconds=round(time.time() - t, 1))
        if "Minf" in out:
            methods[key]["rows_Minf"] = out["Minf"]
        log(f"{name}: {key:<18} done [{methods[key]['seconds']} s]")
    res["methods"] = methods
    finalise_task(res)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def finalise_task(tk):
    """S* rules (from the stored rows). Main rule (the brief): S_star = smallest S such that this and every larger S
    have mean BA >= exact - tol, tol = max(0.01, one test positive = 0.5/n_test_pos). Also stored: E7's 0.01 variants,
    vs the full-memory stream at the same M, and the AUPRC version."""
    ex_ba, ex_ap = tk["exact_centroid"]["balanced_accuracy"], tk["exact_centroid"]["auprc"]
    st_ba = tk["stream_full"]["balanced_accuracy"]
    tol = max(0.01, 0.5 / max(tk["n_test_pos"], 1))
    tk["tolerance"] = tol
    N = tk["N"]
    for key, m in tk["methods"].items():
        rows = m["rows"]
        for nm, target, k, fn, tl in [("S_star", ex_ba, "balanced_accuracy_mean", E7.s_star_stable, tol),
                                      ("S_star_first", ex_ba, "balanced_accuracy_mean", E7.s_star, tol),
                                      ("S_star_tol001", ex_ba, "balanced_accuracy_mean", E7.s_star_stable, 0.01),
                                      ("S_star_tol002", ex_ba, "balanced_accuracy_mean", E7.s_star_stable, max(0.02, tol)),
                                      ("S_star_stream", st_ba, "balanced_accuracy_mean", E7.s_star_stable, tol),
                                      ("S_star_auprc", ex_ap, "auprc_mean", E7.s_star_stable, 0.01)]:
            S, B = fn(rows, target, key=k, tol=tl)
            m[nm] = S
            m[nm + "_over_N"] = (S / N) if S else None
            if nm == "S_star":
                m["bytes_at_S_star"] = B
        if "rows_Minf" in m:
            S, B = E7.s_star_stable(m["rows_Minf"], ex_ba, key="balanced_accuracy_mean", tol=tol)
            m["S_star_Minf"], m["S_star_Minf_over_N"], m["bytes_at_S_star_Minf"] = S, (S / N if S else None), B
    tk["signal"] = bool(ex_ba >= 0.6)          # exact centroid at chance -> S* is not informative
    return tk


ONE_PASS = ("count_sketch", "hashing", "jl_sparse", "heavy_hitters_mg")


def best_one_pass(tk):
    out = {}
    for var, sk, bk in [("stream", "S_star", "bytes_at_S_star"), ("Minf", "S_star_Minf", "bytes_at_S_star_Minf")]:
        c = [(m.get(sk) or 10 * tk["N"], k) for k, m in tk["methods"].items() if k in ONE_PASS]
        S, k = min(c)
        ok = S <= tk["N"]
        out[var] = dict(method=k, S_star=S if ok else None, S_star_over_N=S / tk["N"] if ok else None,
                        bytes=tk["methods"][k].get(bk) if ok else None)
    out.update(out["stream"])          # top level = E7 protocol (stream at M_run)
    return out


# ---------------------------------------------------------------------------------------------------------------
# job entry point (one json per job)
# ---------------------------------------------------------------------------------------------------------------
_CACHE = {}


def job(dataset, feat, K, positive, split, n_seeds, out, test_frac=0.3, min_cells=3, logreg_full=True, rec_cap=3e9):
    t0 = time.time()
    cfg = load_config()
    base_seed = int(cfg["seed"])
    if dataset not in _CACHE:
        _CACHE.clear()
        _CACHE[dataset] = load_dataset(dataset)
    X, genes, labels, summary = _CACHE[dataset]
    N = K if feat == "genes" else K * (K + 1) // 2
    S_grid = [2**e for e in range(3, int(math.floor(math.log2(N))) + 1)]
    if S_grid[-1] != N:
        S_grid.append(N)
    t = time.time()
    d = build(X, genes, labels, positive, feat, K, split, test_frac, min_cells, base_seed)
    build_s = time.time() - t
    ti = TASKS[dataset].index(positive)
    jcfg = dict(dataset=dataset, feature_set=feat, K=K, N=N, positive=positive, split=split, n_seeds=n_seeds, test_frac=test_frac,
                min_cells_per_gene=min_cells, split_seed=(base_seed if split == 0 else base_seed + 1000 * split),
                gene_selection="top-K |mu1-mu0| of single-gene amplitude features, train split only",
                M_over_N_grid=M_GRID, S_grid=S_grid, hash_logreg_max_S=(16384 if N <= 200_000 else 4096),
                logreg="sklearn LogisticRegression(class_weight='balanced', C=1, lbfgs), max_iter 300 (200 if N > 1e6)",
                S_star_rule="smallest S with mean BA within tol of the exact centroid for this and all larger S; tol = max(0.01, 0.5/n_test_pos)",
                bytes_rule="8 B per float64 counter, 4 B per int32 index, 8 B per 64-bit hash seed",
                logreg_full_all_N=logreg_full, record_cap_per_seed=rec_cap, stream="Poissonised uniform stream, importance weights +1/pi_hat, -1/(1-pi_hat); compound-Poisson sampler")
    name = f"{dataset} {feat} K={K} {positive} split{split}"
    r = run_task(name, d, N, S_grid, n_seeds, base_seed * 100000 + K * 10 + ti + 977 * split,
                 hash_logreg_max_S=jcfg["hash_logreg_max_S"], logreg_max_iter=(200 if N > 1e6 else 300),
                 do_logreg_full=logreg_full, m_cap_records=rec_cap)
    r.update(positive=positive, dataset=dataset, feature_set=feat, K=K, split=split, genes_head=d["genes"][:12],
             build_seconds=round(build_s, 1), best_one_pass=best_one_pass(r))
    body = dict(schema_version=1, experiment="e13_job", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                partial=False, config=dict(repo=cfg, job=jcfg), git_commit=git_commit(), hardware=hardware(),
                dataset_summary=summary, wall_seconds=round(time.time() - t0, 1), peak_ram_gb=peak_ram_gb(), peak_vram_gb=0.0, result=r)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(clean(body), separators=(",", ":")))
    b = r["best_one_pass"]
    log(f"{name}: DONE in {body['wall_seconds']:.0f} s, peak RAM {body['peak_ram_gb']} GB; best one-pass {b['method']} S*={b['S_star']} "
        f"S*/N={b['S_star_over_N']}")
    return body


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--feat", required=True, choices=["genes", "pairs"])
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-logreg-full", action="store_true")
    ap.add_argument("--rec-cap", type=float, default=3e9)
    a = ap.parse_args()
    job(a.dataset, a.feat, a.K, a.task, a.split, a.seeds, a.out, logreg_full=not a.no_logreg_full, rec_cap=a.rec_cap)
