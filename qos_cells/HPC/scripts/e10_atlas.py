"""E10 — large atlas with curated labels: AIDA (Asian Immune Diversity Atlas, PBMC), CELLxGENE Census 2025-11-08.

Removes two E7 limits: Louvain-circular labels (AIDA labels are expert annotations mapped to Cell Ontology) and
tiny test sets (all 1,265,624 cells are used: no subsampling, so every prevalence pi is the true atlas value).

Pipeline per task (one-vs-rest), same as E7/E9:
  donor split (30 % of donors -> test, GroupShuffleSplit seed 7) -> top-K genes by |Delta| on TRAIN only
  -> gene features (N = K) and gene-pair features (N = K(K+1)/2) -> exact centroid, class-balanced logreg,
  E7 Poissonised stream (M by convergence check), simulated quantum readout p.sin(c phi), memory-limited sketches.
Plus the rare-class sample cost: records M needed for 10 % relative L2 error of the streamed Delta, measured from
the stream at several M (fit rel_err = c / sqrt(M)), compared with the closed form
    M_10% = 100 N (q1/pi + q0/(1-pi)) / ||Delta||^2      (q_c = mean_i in c of sum_j p_ij^2)
which is the 1/(pi(1-pi)) law times a task constant.

    python e10_atlas.py prep                       # h5ad -> per-task K=512/2048 matrices in HPC/data/e10/
    python e10_atlas.py run --task 0 --space pairs --K 512
    python e10_atlas.py merge
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpc_common as hc  # noqa: E402

TAG = "e10"
H5AD = hc.DATA / "aida_c838aec3.h5ad"
SRC = dict(name="AIDA Phase 1 Data Freeze v2 (Asian Immune Diversity Atlas), PBMC, 10x 5' v2",
           census_version="2025-11-08", dataset_id="c838aec3-03ef-4398-b882-0e3912abfff0",
           collection="Asian Immune Diversity Atlas (AIDA)",
           url="https://cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com/cell-census/2025-11-08/h5ads/c838aec3-03ef-4398-b882-0e3912abfff0.h5ad",
           labels="obs.cell_type (author annotation mapped to Cell Ontology by the submitters; curated, not re-clustered here)")
TASKS = ["CD1c-positive myeloid dendritic cell", "plasma cell", "hematopoietic stem cell", "CD14-positive monocyte"]
TASK_KIND = ["rare (pi ~ 1e-2)", "rare (pi ~ 1e-3)", "rare (pi ~ 1e-4)", "common (pi ~ 0.16)"]
PREP = hc.DATA / "e10"
M_GRID = [33, 100, 300, 900, 2700, 8100, 24300, 72900, 218700, 656100]


def log(msg):
    hc.log(TAG, msg)


# ===================================================================================================================
# prep
# ===================================================================================================================
def prep(args):
    import anndata as ad
    from sklearn.model_selection import GroupShuffleSplit

    tracker = hc.Tracker()
    PREP.mkdir(parents=True, exist_ok=True)
    t = time.time()
    sha = (hc.DATA / "aida_c838aec3.sha256").read_text().split()[0] if (hc.DATA / "aida_c838aec3.sha256").exists() else None
    a = ad.read_h5ad(H5AD)
    log(f"read h5ad {a.shape} in {time.time() - t:.0f} s; raw={a.raw is not None}")
    if a.raw is not None:
        X = sp.csr_matrix(a.raw.X, dtype=np.float32)
        var = a.raw.var
        xsrc = "raw.X (raw counts)"
    else:
        X = sp.csr_matrix(a.X, dtype=np.float32)
        var = a.var
        xsrc = "X"
    is_int = bool(np.allclose(X.data[:100000], np.round(X.data[:100000])))
    log(f"X from {xsrc}, integer counts: {is_int}, nnz {X.nnz:,}")
    if is_int:                                              # same normalisation as PBMC3k processed .raw: CP10k + log1p
        lib = np.asarray(X.sum(1)).ravel()
        lib[lib == 0] = 1
        X = sp.csr_matrix(sp.diags((1e4 / lib).astype(np.float32)) @ X)
        X.data = np.log1p(X.data)
        norm = "counts -> normalise to 1e4 per cell -> log1p (as PBMC3k .raw)"
    else:
        norm = "used as provided (already normalised)"
    genes = np.asarray(var["feature_name"] if "feature_name" in var else var.index, dtype=str)
    labels = np.asarray(a.obs["cell_type"].astype(str))
    donors = np.asarray(a.obs["donor_id"].astype(str))
    n = X.shape[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=7)
    tr, te = next(gss.split(np.zeros(n), groups=donors))
    tr, te = np.sort(tr), np.sort(te)
    assert not set(donors[tr]) & set(donors[te])
    log(f"donor split: train {len(tr):,} cells / {len(set(donors[tr]))} donors, test {len(te):,} / {len(set(donors[te]))}")
    # train-only gene statistics (same rule as qos_cells.load_data.select_genes)
    Xtr = X[tr]
    sq = Xtr.multiply(Xtr).tocsr()
    norm2 = np.asarray(sq.sum(1)).ravel()
    norm2[norm2 == 0] = 1.0
    P = sp.csr_matrix(sp.diags(1.0 / norm2) @ sq)
    n_cells = np.asarray((Xtr > 0).sum(0)).ravel()
    order_rows = np.r_[tr, te]
    Xo = X[order_rows]
    counts = {k: int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    summary = dict(source=SRC, sha256=sha, file_bytes=H5AD.stat().st_size, X_source=xsrc, normalisation=norm,
                   n_cells=int(n), n_genes=int(X.shape[1]), nnz=int(X.nnz), n_donors=int(len(set(donors))),
                   split="by donor (GroupShuffleSplit, test_size 0.3 of donors, seed 7)",
                   n_train=int(len(tr)), n_test=int(len(te)), n_train_donors=int(len(set(donors[tr]))),
                   n_test_donors=int(len(set(donors[te]))), subsampling="none (all cells kept)",
                   class_counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    for ti, pos in enumerate(TASKS):
        y = (labels == pos).astype(int)
        ytr = y[tr]
        delta = np.asarray(P.T @ (ytr / ytr.sum())).ravel() - np.asarray(P.T @ ((1 - ytr) / (1 - ytr).sum())).ravel()
        delta[n_cells < 3] = 0.0
        g = np.argsort(-np.abs(delta), kind="stable")[:2048]
        for K in (512, 2048):
            gk = g[:K]
            sp.save_npz(PREP / f"task{ti}_K{K}_X.npz", sp.csr_matrix(Xo[:, gk]), compressed=False)
        meta = dict(task=pos, kind=TASK_KIND[ti], ntr=int(len(tr)), nte=int(len(te)), y=y[order_rows].tolist(),
                    genes=[str(s) for s in genes[g]], n_pos_total=int(y.sum()), n_pos_train=int(ytr.sum()), n_pos_test=int(y[te].sum()),
                    n_pos_donors_train=int(len(set(donors[tr][ytr == 1]))), n_pos_donors_test=int(len(set(donors[te][y[te] == 1]))),
                    pi_atlas=float(y.mean()))
        (PREP / f"task{ti}.json").write_text(json.dumps(meta))
        log(f"task {ti} {pos}: pos total {meta['n_pos_total']} (train {meta['n_pos_train']}, test {meta['n_pos_test']}), "
            f"pi {meta['pi_atlas']:.2e}, top genes {meta['genes'][:6]}")
    summary["prep_resources"] = tracker.stats()
    (PREP / "summary.json").write_text(json.dumps(hc.clean(summary)))
    log(f"prep done {tracker.stats()}")


# ===================================================================================================================
# gene-feature space with the PairData interface
# ===================================================================================================================
class GeneData:
    def __init__(self, Xsel):
        import cupy as cp
        X = sp.csr_matrix(Xsel, dtype=np.float64)
        X.sort_indices()
        self.n, self.K = X.shape
        self.N = self.K
        sq = X.multiply(X).tocsr()
        nrm = np.asarray(sq.sum(1)).ravel()
        self.abstain = nrm == 0
        inv = np.zeros_like(nrm)
        inv[~self.abstain] = 1.0 / nrm[~self.abstain]
        P = sp.csr_matrix(sp.diags(inv) @ sq)
        P.sort_indices()
        self.q = np.asarray(P.multiply(P).sum(1)).ravel()
        self.npairs = np.diff(P.indptr).astype(np.int64)
        self.Pd = cp.asarray(P.toarray(), dtype=cp.float32)
        self.indptr = cp.asarray(P.indptr.astype(np.int64))
        self.indices = cp.asarray(P.indices.astype(np.int64))
        self.data = cp.asarray(P.data.astype(np.float64))

    def centroid_delta(self, rows, y):
        import cupy as cp
        rows, y = np.asarray(rows), np.asarray(y)
        r1, r0 = cp.asarray(rows[y == 1]), cp.asarray(rows[y == 0])
        return self.Pd[r1].astype(cp.float64).mean(0) - self.Pd[r0].astype(cp.float64).mean(0)

    def scores(self, D):
        import cupy as cp
        D = cp.asarray(D, dtype=cp.float64)
        mx = float(cp.abs(D).max()) or 1.0
        return (self.Pd @ (D / mx).astype(cp.float32)).astype(cp.float64) * mx

    def stream(self, rows, w_cell, M, seed, batch=50_000_000, n_records_override=None):
        """rows must be 0..ntr-1 (train cells first). Poissonised uniform stream over (cell, gene) entries."""
        import cupy as cp
        ntr = len(rows)
        assert rows[0] == 0 and rows[-1] == ntr - 1
        E = int(self.indptr[ntr])
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
        R = int(rng.poisson(M / (self.N * ntr) * E)) if n_records_override is None else int(n_records_override)
        rs = cp.random.RandomState(int(rng.integers(0, 2**31 - 1)))
        w = cp.asarray(w_cell, dtype=cp.float64)
        pos = cp.zeros(self.N)
        neg = cp.zeros(self.N)
        done = 0
        while done < R:
            B = min(batch, R - done)
            k = cp.minimum(cp.floor(rs.random_sample(B) * E).astype(cp.int64), E - 1)
            row = cp.searchsorted(self.indptr, k, side="right") - 1
            v = w[row] * self.data[k]
            g = self.indices[k]
            ip = v > 0
            pos += cp.bincount(g[ip], weights=v[ip], minlength=self.N)
            neg += cp.bincount(g[~ip], weights=-v[~ip], minlength=self.N)
            done += B
        if n_records_override is not None:
            return pos - neg, pos, R
        return (self.N / M) * (pos - neg), (self.N / M) * pos, R


# ===================================================================================================================
# run one (task, space)
# ===================================================================================================================
def run(args):
    import cupy as cp
    from qos_cells.readout import best_threshold, evaluate
    from qos_cells.sketch import rescale_safe
    from qos_cells.stream import class_weights
    from pairs_gpu import (MG, PairData, TopS, d_eff, hashed_features, ledger, logreg_dense, logreg_pairs,
                           sk_count_sketch, sk_hashing, sk_jl_sparse)
    import e9_fullpairs as e9

    E7 = e9.e7mod()
    tracker = hc.Tracker()
    meta = json.loads((PREP / f"task{args.task}.json").read_text())
    summary = json.loads((PREP / "summary.json").read_text())
    y = np.asarray(meta["y"])
    ntr = meta["ntr"]
    ytr, yte = y[:ntr], y[ntr:]
    X = sp.load_npz(PREP / f"task{args.task}_K{args.K}_X.npz")
    t0 = time.time()
    data = PairData(X) if args.space == "pairs" else GeneData(X)
    N = data.N
    out_name = f"e10_part_task{args.task}_{args.space}_K{args.K}.json"
    exp_cfg = dict(task=meta["task"], kind=meta["kind"], space=args.space, K=args.K, seeds=args.seeds, M_over_N_grid=M_GRID,
                   max_records_per_seed=args.max_records, time_cap_min=args.time_cap, hashing_logreg_S=args.hl_S,
                   dataset=summary)
    log(f"{meta['task']} {args.space} K={args.K}: N={N:,}, n={data.n:,} built in {time.time() - t0:.0f} s")
    ab_tr, ab_te = data.abstain[:ntr], data.abstain[ntr:]

    def ev_scores(s):
        s = np.asarray(s, dtype=np.float64)
        a, b = s[:ntr].copy(), s[ntr:].copy()
        a[ab_tr] = -np.inf
        b[ab_te] = -np.inf
        thr = best_threshold(a, ytr)
        e = evaluate(b, yte, thr)
        return dict(balanced_accuracy=e["balanced_accuracy"], auprc=e["auprc"])

    def ev(D):
        return ev_scores(cp.asnumpy(data.scores(D)))

    tr_rows = np.arange(ntr)
    pi_hat = float(ytr.mean())
    w_cell = class_weights(pi_hat)[ytr]
    E_tr = int(data.npairs[:ntr].sum())
    res = dict(task=meta["task"], kind=meta["kind"], space=args.space, K=args.K, N=int(N), index_qubits=int(math.ceil(math.log2(N))),
               n_train=int(ntr), n_test=int(len(yte)), n_train_pos=int(ytr.sum()), n_test_pos=int(yte.sum()),
               n_pos_donors_train=meta["n_pos_donors_train"], n_pos_donors_test=meta["n_pos_donors_test"],
               pi_hat=pi_hat, pi_atlas=meta["pi_atlas"], base_rate_test=float(yte.mean()),
               abstain_rate_test=float(ab_te.mean()), abstain_rate_test_pos=float(ab_te[yte == 1].mean()),
               train_density=E_tr / (ntr * N), nnz_train=E_tr, genes_head=meta["genes"][:12])

    def dump(partial=True):
        res["seconds"] = round(time.time() - t0, 1)
        hc.write_json(out_name, exp_cfg, dict(task=res), tracker, partial=partial)

    delta = data.centroid_delta(tr_rows, ytr)
    e = ev(delta)
    res["exact_centroid"] = dict(**e, **ledger(floats=N))
    res["d_eff"] = d_eff(delta)
    res["delta_norm2"] = float(cp.sum(delta**2))
    exact_ba = e["balanced_accuracy"]
    q1 = float(data.q[:ntr][ytr == 1].mean())
    q0 = float(data.q[:ntr][ytr == 0].mean())
    res["q_pos"], res["q_neg"] = q1, q0
    res["M10_closed_form"] = 100 * N * (q1 / pi_hat + q0 / (1 - pi_hat)) / res["delta_norm2"]
    log(f"exact bal.acc {exact_ba:.4f} auprc {e['auprc']:.4f} d_eff {res['d_eff']:.1f} pi {pi_hat:.2e} "
        f"test pos {res['n_test_pos']} M10 closed form {res['M10_closed_form']:.3e} (M/N {res['M10_closed_form'] / N:.3g})")
    dump()
    # ---- logistic regression (class-balanced, all N features) ---------------------------------------------------
    t = time.time()
    scale = E_tr / max(1, int((~ab_tr).sum()))
    if args.space == "pairs":
        w, b, nit, convd = logreg_pairs(data, tr_rows, ytr, scale, max_iter=300)
        z = scale * data.scores(cp.asarray(w)) + b
    else:
        w, b, nit, convd = logreg_dense(data.Pd[:ntr], ytr, scale, max_iter=300)
        z = scale * (data.Pd @ cp.asarray(w, dtype=cp.float32)).astype(cp.float64) + b
    e = ev_scores(cp.asnumpy(z))
    res["logreg_full"] = dict(**e, **ledger(floats=N + 1), n_iter=nit, converged=convd, seconds=round(time.time() - t, 1))
    log(f"logreg_full bal.acc {e['balanced_accuracy']:.4f} auprc {e['auprc']:.4f} iters {nit} [{time.time() - t:.0f} s]")
    dump()
    # ---- stream, M by convergence check ---------------------------------------------------------------------------
    pred = lambda mo: math.sqrt(N / (mo * N * ntr) * float(np.sum(w_cell**2 * data.q[:ntr]))) / math.sqrt(res["delta_norm2"])
    start = next((i for i, mo in enumerate(M_GRID) if pred(mo) <= args.start_rel_err), len(M_GRID) - 1)
    rec_per_MN = E_tr / ntr                                       # expected records per unit of M/N
    msel, phis, conv_at = [], None, None
    for mo in M_GRID[max(0, start - 1):]:
        if mo * rec_per_MN > args.max_records:
            log(f"M/N={mo}: {mo * rec_per_MN:.2e} records/seed > cap {args.max_records:.1e}; stop (cut logged)")
            res["M_cap_hit"] = dict(M_over_N=mo, records_per_seed=mo * rec_per_MN, cap=args.max_records)
            break
        if (time.time() - t0) / 60 > args.time_cap * 0.6 and phis is not None:
            log("time cap for M selection reached; stop (cut logged)")
            res["M_time_cap_hit"] = True
            break
        M = int(mo * N)
        cur, evs, qs = [], [], []
        for s in range(args.seeds):
            phi, ppos, R = data.stream(tr_rows, w_cell, M, seed=1000 * args.task + 10 * s + M_GRID.index(mo))
            cur.append((phi, ppos, R))
            evs.append(ev(phi))
            ps, c = rescale_safe(cp.asnumpy(phi))
            qs.append(ev(cp.sin(cp.asarray(ps))))
        rel = [float(cp.linalg.norm(c[0] - delta) / cp.linalg.norm(delta)) for c in cur]
        row = dict(M_over_N=mo, M=M, nonzero_records_mean=float(np.mean([c[2] for c in cur])),
                   balanced_accuracy_mean=float(np.mean([x["balanced_accuracy"] for x in evs])),
                   balanced_accuracy_std=float(np.std([x["balanced_accuracy"] for x in evs])),
                   auprc_mean=float(np.mean([x["auprc"] for x in evs])), auprc_std=float(np.std([x["auprc"] for x in evs])),
                   rel_error_mean=float(np.mean(rel)), rel_error_seeds=rel, rel_error_predicted=pred(mo),
                   quantum_sin_readout_balanced_accuracy_mean=float(np.mean([x["balanced_accuracy"] for x in qs])),
                   quantum_sin_readout_auprc_mean=float(np.mean([x["auprc"] for x in qs])))
        msel.append(row)
        phis = cur
        log(f"M/N={mo} stream bal.acc {row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f} auprc {row['auprc_mean']:.3f} "
            f"rel.err {row['rel_error_mean']:.3f} (pred {row['rel_error_predicted']:.3f}) records {row['nonzero_records_mean']:.2e}")
        res["M_selection"] = msel
        dump()
        if conv_at is not None:
            break
        if row["balanced_accuracy_mean"] >= exact_ba - 0.01 and row["rel_error_mean"] < 1.0:
            conv_at = mo
    run_row = msel[-1]
    res["M_converged_over_N"] = conv_at
    res["M_run"], res["M_run_over_N"] = run_row["M"], run_row["M_over_N"]
    res["M_rule"] = ("start at the grid M one step below the first with predicted rel. error <= %.1f; converged = stream bal.acc >= exact - 0.01 "
                     "and rel. error < 1; run one grid step beyond; capped by records/seed and time" % args.start_rel_err)
    res["stream_full"] = dict(balanced_accuracy=run_row["balanced_accuracy_mean"], auprc=run_row["auprc_mean"],
                              balanced_accuracy_std=run_row["balanced_accuracy_std"], **ledger(floats=N),
                              nonzero_records_mean=run_row["nonzero_records_mean"])
    res["quantum_sim"] = dict(index_qubits=res["index_qubits"], balanced_accuracy=run_row["quantum_sin_readout_balanced_accuracy_mean"],
                              auprc=run_row["quantum_sin_readout_auprc_mean"], M=run_row["M"],
                              note="classical simulation of the oracle sketch (p.sin(c phi)); NOT run on hardware")
    # sample cost: fit rel_err = c / sqrt(M) over all measured M (slope fixed at -1/2)
    Ms = np.array([r["M"] for r in msel], float)
    re = np.array([r["rel_error_mean"] for r in msel], float)
    c = float(np.exp(np.mean(np.log(re) + 0.5 * np.log(Ms))))
    slope = float(np.polyfit(np.log(Ms), np.log(re), 1)[0]) if len(Ms) > 1 else None
    res["sample_cost"] = dict(M10_measured=(c / 0.1) ** 2, M10_closed_form=res["M10_closed_form"],
                              ratio_measured_over_closed=(c / 0.1) ** 2 / res["M10_closed_form"], fitted_slope=slope,
                              law_constant_kappa=N * (q1 * (1 - pi_hat) + q0 * pi_hat) / res["delta_norm2"],
                              M10_times_pi_1mpi_over_N=(c / 0.1) ** 2 * pi_hat * (1 - pi_hat) / N,
                              rule="rel_err(M) = ||phi_M - Delta|| / ||Delta||, mean over seeds; M10 = M with rel_err = 0.1 from the c/sqrt(M) fit")
    dump()
    # ---- memory-limited methods -----------------------------------------------------------------------------------
    S_grid = [2**e for e in range(3, int(math.floor(math.log2(N))) + 1)]
    if S_grid[-1] != N:
        S_grid.append(N)
    res["S_grid"] = S_grid
    jidx = cp.arange(N, dtype=cp.uint64)
    specs = [("count_sketch", "count-sketch (3 rows, median)"), ("hashing", "feature hashing (signed, 1 row)"),
             ("jl_sparse", "sparse JL projection (4 rows)"), ("heavy_hitters_mg", "heavy hitters: Misra–Gries (S/2 + S/2)"),
             ("heavy_hitters_top", "heavy hitters: oracle top-S |Δ|"), ("reservoir", "reservoir: S nonzero records")]
    if args.space == "pairs" and args.hl_S:
        specs.append(("hashing_logreg", "feature hashing + logistic regression"))
    methods = {}
    for mk, (key, label) in enumerate(specs):
        t = time.time()
        per_seed = []
        n_s = 1 if key == "hashing_logreg" else args.seeds
        for s in range(n_s):
            phi, ppos, _ = phis[s]
            rng = np.random.default_rng(np.random.SeedSequence([7, args.task, 11, s, mk, args.K]))
            seeds = [int(v) for v in rng.integers(0, 2**62, size=16)]
            metrics = {}
            if key == "hashing_logreg":
                grid = [S for S in S_grid if S in args.hl_S]
                H = hashed_features(data, np.arange(data.n), grid, seeds)
                for S in grid:
                    w, b, nit, convd = logreg_dense(H[S][:ntr], ytr, scale, max_iter=200)
                    z = scale * (H[S] @ cp.asarray(w, dtype=cp.float32)).astype(cp.float64) + b
                    metrics[S] = dict(**ev_scores(cp.asnumpy(z)), mem=ledger(floats=S + 1, seeds=2), n_iter=nit)
                del H
                per_seed.append(metrics)
                continue
            if key == "heavy_hitters_mg":
                mg = MG(ppos, ppos - phi)
            elif key == "heavy_hitters_top":
                top = TopS(phi)
            for S in S_grid:
                if key == "count_sketch":
                    d, m = sk_count_sketch(phi, S, seeds, jidx)
                elif key == "hashing":
                    d, m = sk_hashing(phi, S, seeds, jidx)
                elif key == "jl_sparse":
                    d, m = sk_jl_sparse(phi, S, seeds, jidx)
                elif key == "heavy_hitters_mg":
                    d, m = mg(S)
                elif key == "heavy_hitters_top":
                    d, m = top(S)
                elif key == "reservoir":
                    d, _, _ = data.stream(tr_rows, w_cell, None, seed=seeds[0] % (2**31) + S, n_records_override=S)
                    m = ledger(floats=S, ints=S)
                metrics[S] = dict(**ev(d), mem=m)
            per_seed.append(metrics)
            cp.get_default_memory_pool().free_all_blocks()
        rows = E7.summarise(per_seed)
        Sst, Bst = E7.s_star(rows, exact_ba)
        methods[key] = dict(label=label, rows=rows, seeds=n_s, S_star=Sst, S_star_over_N=(Sst / N if Sst else None),
                            bytes_at_S_star=Bst, seconds=round(time.time() - t, 1))
        res["methods"] = methods
        log(f"{key:<18} S* = {Sst} [{methods[key]['seconds']} s]")
        dump()
    # S* rules exactly as E7 (_finalise): stable, cell-granularity, vs stream, AUPRC
    sub = dict(K=args.K, real=[res], synthetic=None, quantum=dict(index_qubits=res["index_qubits"]))
    E7._finalise([sub])
    for k, m in res["methods"].items():
        log(f"  {k:<18} S*_stable {m['S_star_stable']}  S*_cell_stable {m['S_star_cell_stable']}")
    dump(partial=False)
    log(f"done {tracker.stats()}")


# ===================================================================================================================
# merge
# ===================================================================================================================
def merge(args):
    parts = sorted(hc.RESULTS.glob("e10_part_task*_*.json"))
    tasks, meta = [], {}
    for p in parts:
        d = json.loads(p.read_text())
        meta[p.name] = dict(partial=d["partial"], resources=d["resources"], git_commit=d["git_commit"])
        tasks.append(d["task"])
    summary = json.loads((PREP / "summary.json").read_text())
    deff_points, table = [], []
    for t in tasks:
        best = None
        if "methods" in t and all("S_star_cell_stable" in m for m in t["methods"].values()):
            cands = []
            for k, m in t["methods"].items():
                if k in ("hashing_logreg", "heavy_hitters_top"):
                    continue
                v = m.get("S_star_cell_stable_over_N")
                cands.append((v if v is not None else 2.0, k))
            v, k = min(cands)
            best = dict(method=k, S_star_over_N=(v if v <= 1 else None), S_star=(round(v * t["N"]) if v <= 1 else None))
            t["best_one_pass"] = best
        deff_points.append(dict(label=f"AIDA {t['task']} ({t['space']})", dataset=f"AIDA (E10, {t['space']})", task=t["task"], K=t["K"],
                                N=t["N"], d_eff=t["d_eff"], best_S_star_over_N=(best or {}).get("S_star_over_N")))
        table.append(dict(task=t["task"], space=t["space"], N=t["N"], pi=t["pi_hat"], n_test_pos=t["n_test_pos"],
                          exact_ba=t["exact_centroid"]["balanced_accuracy"], exact_auprc=t["exact_centroid"]["auprc"],
                          logreg_ba=t.get("logreg_full", {}).get("balanced_accuracy"), logreg_auprc=t.get("logreg_full", {}).get("auprc"),
                          stream_ba=t.get("stream_full", {}).get("balanced_accuracy"), quantum_sim_ba=t.get("quantum_sim", {}).get("balanced_accuracy"),
                          quantum_sim_auprc=t.get("quantum_sim", {}).get("auprc"), M_run_over_N=t.get("M_run_over_N"),
                          best_method=(best or {}).get("method"), best_S_star=(best or {}).get("S_star"),
                          best_S_star_over_N=(best or {}).get("S_star_over_N"), d_eff=t["d_eff"],
                          M10_measured=t.get("sample_cost", {}).get("M10_measured"), M10_closed=t.get("M10_closed_form")))
    payload = dict(dataset=summary, tasks=tasks, summary_table=table, deff_points=deff_points, parts=meta,
                   limits=["one atlas (AIDA, PBMC, 10x 5'), one donor split",
                           "labels are curated annotations, but still produced by the atlas authors from the same expression data plus expert review",
                           "stream is Poissonised (Poisson splitting), not record-by-record",
                           "quantum column = classical simulation of the sketch; nothing run on hardware",
                           "S counts counters; a qubit and a float are not the same unit"])
    partial = any(m["partial"] for m in meta.values()) or len(tasks) < 2 * len(TASKS)
    hc.write_json("e10_atlas.json", dict(merged_from=sorted(meta)), payload, hc.Tracker(), partial=partial)
    log(f"merged {len(parts)} parts (partial={partial})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    r = sub.add_parser("run")
    r.add_argument("--task", type=int, required=True)
    r.add_argument("--space", choices=["genes", "pairs"], default="pairs")
    r.add_argument("--K", type=int, default=512)
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--max-records", type=float, default=4e9)
    r.add_argument("--time-cap", type=float, default=40.0)
    r.add_argument("--start-rel-err", type=float, default=2.5)
    r.add_argument("--hl-S", type=int, nargs="*", default=[64, 256, 1024])
    sub.add_parser("merge")
    a = ap.parse_args()
    {"prep": prep, "run": run, "merge": merge}[a.cmd](a)
