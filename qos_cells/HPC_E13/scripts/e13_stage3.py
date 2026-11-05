"""E13 Stage 3 — PBMC68k gene-pair space at K = 8192 (N = 3.4e7) and all expressed genes (K ~ 17k, N ~ 1.5e8), GPU.

Reuses HPC/scripts/pairs_gpu.py (E9) unchanged: per-cell pair vectors are never built.
  * class centroid of pair features = upper triangle of a scaled Gram matrix  U^T U / n  with  u = x^2 / sqrt(Z),
    Z = sum_{g<=h} x_g^2 x_h^2 = ((sum x^2)^2 + sum x^4) / 2;
  * any linear score sum_j p_ij D_j = u_i^T T u_i  (T = D on the upper triangle), computed in row blocks;
  * the stream is E7's Poissonised uniform stream sampled by Poisson splitting on the GPU.
Gate (--K 512): the GPU path is compared with direct materialisation (e13_core.pair_features_fast == E7 pair features):
Z, Delta, and the exact-centroid balanced accuracy.

Same tasks, split (split 0 = config seed), train-only gene rule, methods, S* rule (stable, tol = max(0.01, one test
positive)) and JSON schema as Stage 1. Stage-3 cuts (recorded in the JSON): S grid = powers of 2 from 8 to 2^22 plus N
(as E9); the M->inf variant uses 1 seed; hashing+logreg 1 seed, S <= 2^14, only where --hl is given.

    python e13_stage3.py job --K 8192 --task "CD34+"           (one task, one GPU)
    python e13_stage3.py job --K all  --task "CD34+"
    python e13_stage3.py merge
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e13_core as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "HPC" / "scripts"))
PARTS = C.E13 / "logs" / "parts_stage3"
REC_CAP = 3e10                    # records per seed per M value (GPU sampler ~3e8 records/s); --rec-cap overrides


def _patch_lbfgs():
    """scipy's L-BFGS-B workspace (2m+5)*n is int32-indexed: E9 uses m = 5 above n = 5e7, which still overflows at
    n > 1.4e8 (all-genes pairs, N = 1.47e8). Same objective and solver, m = 3 there. Patched here, pairs_gpu.py unchanged."""
    import pairs_gpu
    from scipy.optimize import minimize
    orig = pairs_gpu.fit_logreg

    def fit(forward, backward, n_w, y, C=1.0, max_iter=300, tol=1e-4):
        if (2 * 5 + 5) * (n_w + 1) < 2**31 - 1:
            return orig(forward, backward, n_w, y, C=C, max_iter=max_iter, tol=tol)
        import scipy.optimize as so
        real = so.minimize

        def m3(*args, **kw):
            kw["options"] = dict(kw["options"], maxcor=3)
            return real(*args, **kw)
        so.minimize = m3
        try:
            return orig(forward, backward, n_w, y, C=C, max_iter=max_iter, tol=tol)
        finally:
            so.minimize = real
    pairs_gpu.fit_logreg = fit


def s_grid_for(N):
    top = min(22, int(math.floor(math.log2(N))))
    g = [2**e for e in range(3, top + 1)]
    if g[-1] != N:
        g.append(N)
    return g


def peak_vram_gb():
    import cupy as cp
    try:
        return round(cp.cuda.Device().mem_info[1] / 2**30 - cp.cuda.Device().mem_info[0] / 2**30, 2)
    except Exception:
        return None


class VramTracker:
    def __init__(self):
        self.peak = 0.0

    def tick(self):
        import cupy as cp
        used = cp.get_default_memory_pool().total_bytes() / 2**30
        self.peak = max(self.peak, used)


def job(a):
    import cupy as cp
    from qos_cells.readout import best_threshold, evaluate
    from qos_cells.sketch import rescale_safe
    from pairs_gpu import MG, PairData, TopS, d_eff, hashed_features, ledger, logreg_dense, logreg_pairs, sk_count_sketch, sk_hashing, sk_jl_sparse

    global REC_CAP
    REC_CAP = a.rec_cap
    _patch_lbfgs()
    t0 = time.time()
    vt = VramTracker()
    cfg = C.load_config()
    base = int(cfg["seed"])
    X, genes, labels, summary = C.load_dataset(a.dataset)
    pos = a.task
    ti = C.TASKS[a.dataset].index(pos)
    y = (labels == pos).astype(int)
    tr, te = C.split_indices(y, a.split, 0.3, base)
    ytr, yte = y[tr], y[te]
    nc = np.asarray((X[tr] > 0).sum(0)).ravel()
    K = int((nc >= 3).sum()) if a.K == "all" else int(a.K)
    g = C.select_genes(X[tr], ytr, K, 3)                                    # TRAIN ONLY
    rows = np.r_[tr, te]
    ntr = len(tr)
    tb = time.time()
    pd = PairData(X[rows][:, g])
    N = pd.N
    build_s = time.time() - tb
    S_grid = s_grid_for(N)
    pi_hat = float(ytr.mean())
    w_cell = C.class_weights(pi_hat)[ytr]
    tr_rows = np.arange(ntr)
    ab_tr, ab_te = pd.abstain[:ntr], pd.abstain[ntr:]
    E_tr = int(pd.npairs[:ntr].sum())

    def ev_scores(s):
        s = np.asarray(s, dtype=np.float64)
        x, z = s[:ntr].copy(), s[ntr:].copy()
        x[ab_tr] = -np.inf
        z[ab_te] = -np.inf
        e = evaluate(z, yte, best_threshold(x, ytr))
        return dict(balanced_accuracy=e["balanced_accuracy"], auprc=e["auprc"])

    def ev(D):
        vt.tick()
        return ev_scores(cp.asnumpy(pd.scores(D)))

    name = f"{a.dataset} pairs K={K} {pos} split{a.split}"
    res = dict(task=name, dataset=a.dataset, feature_set="pairs", K=K, K_rule=("all genes detected in >= 3 train cells" if a.K == "all" else "top-K"),
               positive=pos, split=a.split, N=int(N), index_qubits=int(math.ceil(math.log2(N))), logical_qubits_approx=2 * int(math.ceil(math.log2(N))),
               n_train=int(ntr), n_test=int(len(te)), n_train_pos=int(ytr.sum()), n_test_pos=int(yte.sum()), pi_hat=pi_hat,
               base_rate_test=float(yte.mean()), abstain_rate_test=float(ab_te.mean()), train_density=E_tr / (ntr * N),
               nnz_train=E_tr, nnz_test=int(pd.npairs[ntr:].sum()), genes_head=[str(s) for s in genes[g][:12]], S_grid=S_grid,
               build_seconds=round(build_s, 1))
    # ---- exact centroid -------------------------------------------------------------------------------------
    r1, r0 = tr_rows[ytr == 1], tr_rows[ytr == 0]
    mu1 = pd.from_T(pd.gram(r1) / len(r1))
    mu0 = pd.from_T(pd.gram(r0) / len(r0))
    delta = mu1 - mu0
    e = ev(delta)
    res["exact_centroid"] = dict(**e, **ledger(floats=N))
    res["d_eff"] = d_eff(delta)
    res["d_eff_over_N"] = res["d_eff"] / N
    exact_ba = e["balanced_accuracy"]
    C.log(f"{name}: N={N:,} exact BA {exact_ba:.4f} AUPRC {e['auprc']:.4f} d_eff {res['d_eff']:.0f} (build {build_s:.0f} s, train nnz {E_tr:.3e})")
    # ---- gate: compare with direct materialisation ------------------------------------------------------------
    if a.gate:
        P_tr, abm = C.pair_features_fast(X[tr][:, g])
        ind1 = (ytr == 1).astype(np.float64) / ytr.sum()
        ind0 = (ytr == 0).astype(np.float64) / (ytr == 0).sum()
        d_cpu = np.asarray(P_tr.T @ ind1) - np.asarray(P_tr.T @ ind0)
        d_gpu = cp.asnumpy(delta)
        P_te, _ = C.pair_features_fast(X[te][:, g])
        ev_cpu = C.evaluate_columns(P_tr, ytr, P_te, yte, d_cpu[:, None], abm, ab_te)[0]
        # Z check: sum of the materialised (unnormalised) pair values vs the closed form
        xs = X[rows][:, g].tocsr()
        sq = xs.multiply(xs).tocsr()
        Zc = (np.asarray(sq.sum(1)).ravel() ** 2 + np.asarray(sq.multiply(sq).sum(1)).ravel()) / 2
        Zd = np.array([sum(sq.data[sq.indptr[i]:sq.indptr[i + 1]][a_] * sq.data[sq.indptr[i]:sq.indptr[i + 1]][b_]
                           for a_, b_ in zip(*np.triu_indices(sq.indptr[i + 1] - sq.indptr[i]))) for i in range(200)])
        res["gate_vs_materialised"] = dict(
            max_abs_delta_diff=float(np.abs(d_cpu - d_gpu).max()), max_abs_delta=float(np.abs(d_cpu).max()),
            rel_l2_delta_diff=float(np.linalg.norm(d_cpu - d_gpu) / np.linalg.norm(d_cpu)),
            exact_BA_materialised=ev_cpu["balanced_accuracy"], exact_BA_gram=exact_ba,
            exact_AUPRC_materialised=ev_cpu["auprc"], exact_AUPRC_gram=e["auprc"],
            Z_closed_form_vs_direct_max_rel_diff_first200=float(np.max(np.abs(Zc[:200] - Zd) / np.maximum(Zd, 1e-300))),
            abstain_equal=bool((abm == ab_tr).all()))
        res["gate_vs_materialised"]["pass"] = bool(res["gate_vs_materialised"]["rel_l2_delta_diff"] < 1e-5 and
                                                   abs(ev_cpu["balanced_accuracy"] - exact_ba) < 1e-9)
        C.log(f"{name}: gate {res['gate_vs_materialised']}")
        del P_tr, P_te
    # ---- logreg on all N pair features ------------------------------------------------------------------------
    scale = E_tr / max(1, int((~ab_tr).sum()))
    if not a.no_logreg:
        t = time.time()
        w, b, nit, convd = logreg_pairs(pd, tr_rows, ytr, scale, max_iter=a.logreg_iter)
        z = scale * pd.scores(cp.asarray(w)) + b
        el = ev_scores(cp.asnumpy(z))
        res["logreg_full"] = dict(**el, **ledger(floats=N + 1), feature_scale=scale, n_iter=nit, converged=convd, max_iter=a.logreg_iter,
                                  seconds=round(time.time() - t, 1), method="class-balanced L2 logreg (C=1) on all N pairs, K x K form, GPU L-BFGS (E9)")
        C.log(f"{name}: logreg all N BA {el['balanced_accuracy']:.4f} AUPRC {el['auprc']:.4f} it {nit} [{time.time() - t:.0f} s]")
        del w
    if a.measure_only:
        res["seconds"] = round(time.time() - t0, 1)
        return _write(a, res, summary, t0, vt, partial=True)
    # ---- M selection ------------------------------------------------------------------------------------------
    msel, phis, conv_i = [], None, None
    dn = float(cp.linalg.norm(delta))
    for mi, mo in enumerate(C.M_GRID):
        M = int(mo * N)
        if M / (N * ntr) * E_tr > REC_CAP:
            C.log(f"{name}: M/N={mo} needs > {REC_CAP:.0e} records per seed; stop")
            break
        cur, evs, qs = [], [], []
        for s in range(a.seeds):
            phi, ppos, R = pd.stream(tr_rows, w_cell, M, seed=(base * 7919 + K * 31 + ti * 101 + s * 13 + mi + 1000003 * a.split) % (2**31))
            cur.append((phi, ppos, R))
            evs.append(ev(phi))
            ps, _ = rescale_safe(cp.asnumpy(phi))
            qs.append(ev(cp.sin(cp.asarray(ps))))
        rel = [float(cp.linalg.norm(c[0] - delta)) / dn for c in cur]
        row = dict(M_over_N=mo, M=M, nonzero_records_mean=float(np.mean([c[2] for c in cur])),
                   balanced_accuracy_mean=float(np.mean([x["balanced_accuracy"] for x in evs])),
                   balanced_accuracy_std=float(np.std([x["balanced_accuracy"] for x in evs])),
                   auprc_mean=float(np.mean([x["auprc"] for x in evs])), balanced_accuracy_seeds=[round(x["balanced_accuracy"], 4) for x in evs],
                   rel_error_mean=float(np.mean(rel)),
                   quantum_sin_readout_balanced_accuracy_mean=float(np.mean([x["balanced_accuracy"] for x in qs])),
                   quantum_sin_readout_auprc_mean=float(np.mean([x["auprc"] for x in qs])))
        msel.append(row)
        phis = cur
        C.log(f"{name}: M/N={mo} stream BA {row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f} rel.err {row['rel_error_mean']:.3f} "
              f"records {row['nonzero_records_mean']:.2e}")
        if conv_i is not None:
            break
        if row["balanced_accuracy_mean"] >= exact_ba - 0.01 and row["rel_error_mean"] < 1:
            conv_i = mi
    run_row = msel[-1]
    res.update(M_selection=msel, M_converged_over_N=(C.M_GRID[conv_i] if conv_i is not None else None), M_run=run_row["M"],
               M_run_over_N=run_row["M_over_N"], M_rule="as Stage 1; record cap 3e10 per seed")
    res["stream_full"] = dict(balanced_accuracy=run_row["balanced_accuracy_mean"], auprc=run_row["auprc_mean"],
                              balanced_accuracy_std=run_row["balanced_accuracy_std"], **ledger(floats=N),
                              nonzero_records_mean=run_row["nonzero_records_mean"], rel_error_mean=run_row["rel_error_mean"])
    res["quantum_sim"] = dict(index_qubits=res["index_qubits"], logical_qubits_approx=res["logical_qubits_approx"],
                              balanced_accuracy=run_row["quantum_sin_readout_balanced_accuracy_mean"], auprc=run_row["quantum_sin_readout_auprc_mean"],
                              M=run_row["M"], note="classical simulation of the oracle sketch; not run on hardware")
    # ---- memory-limited methods -------------------------------------------------------------------------------
    jidx = cp.arange(N, dtype=cp.uint64)
    methods = {}
    for mk, (key, label) in enumerate(C.METHODS):
        if key == "hashing_logreg" and not a.hl:
            continue
        t = time.time()
        out = {}
        for src in (["stream"] if key == "hashing_logreg" else ["stream", "Minf"]):
            per_seed = []
            n_s = a.seeds if (src == "stream" and key != "hashing_logreg") else (a.hl_seeds if key == "hashing_logreg" else a.minf_seeds)
            for s in range(n_s):
                if src == "stream":
                    phi, ppos, _ = phis[s]
                    pneg = ppos - phi
                else:
                    phi, ppos, pneg = delta, mu1, mu0
                rng = np.random.default_rng(np.random.SeedSequence([base, K, 11, ti, s, mk] + ([a.split] if a.split else [])))
                seeds = [int(v) for v in rng.integers(0, 2**62, size=16)]
                metrics = {}
                if key == "hashing_logreg":
                    grid = [S for S in S_grid if S <= 2**14]
                    H = hashed_features(pd, np.arange(pd.n), grid, seeds)
                    for S in grid:
                        w, b, nit, _ = logreg_dense(H[S][:ntr], ytr, scale, max_iter=a.logreg_iter)
                        z = scale * (H[S] @ cp.asarray(w, dtype=cp.float32)).astype(cp.float64) + b
                        metrics[S] = dict(**ev_scores(cp.asnumpy(z)), mem=ledger(floats=S + 1, seeds=2))
                    del H
                    per_seed.append(metrics)
                    continue
                if key == "heavy_hitters_mg":
                    mg = MG(ppos, pneg)
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
                    else:
                        d, m = top(S)
                    metrics[S] = dict(**ev(d), mem=m)
                    del d
                mg = top = None
                per_seed.append(metrics)
                cp.get_default_memory_pool().free_all_blocks()
            out[src] = C.E7.summarise(per_seed)
        methods[key] = dict(label=label, rows=out["stream"], seconds=round(time.time() - t, 1), seeds=(a.hl_seeds if key == "hashing_logreg" else a.seeds))
        if "Minf" in out:
            methods[key]["rows_Minf"] = out["Minf"]
            methods[key]["seeds_Minf"] = a.minf_seeds
        res["methods"] = methods
        C.finalise_task(res)
        C.log(f"{name}: {key:<18} S* = {methods[key]['S_star']} (Minf {methods[key].get('S_star_Minf')}) [{methods[key]['seconds']} s]")
        res["seconds"] = round(time.time() - t0, 1)
        _write(a, res, summary, t0, vt, partial=True)
    res["best_one_pass"] = C.best_one_pass(res)
    res["seconds"] = round(time.time() - t0, 1)
    return _write(a, res, summary, t0, vt, partial=False)


def _write(a, res, summary, t0, vt, partial):
    jcfg = dict(stage=(1 if a.out else 3), path="GPU (pairs_gpu.py)", dataset=a.dataset, feature_set="pairs", K_arg=a.K, task=a.task, split=a.split, seeds=a.seeds, seeds_Minf=a.minf_seeds,
                hashing_logreg=bool(a.hl), hashing_logreg_seeds=a.hl_seeds, record_cap_note="see record_cap_per_seed", hashing_logreg_max_S=2**14, logreg_full=not a.no_logreg,
                logreg_max_iter=a.logreg_iter, M_over_N_grid=C.M_GRID, record_cap_per_seed=REC_CAP, S_grid="powers of 2 from 8 to 2^22, plus N",
                gpu="1 x RTX A6000 (CUDA device 0 of the visible set)", method_code="HPC/scripts/pairs_gpu.py (E9), unchanged")
    body = dict(schema_version=1, experiment="e13_stage3_job", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                partial=partial, config=dict(repo=C.load_config(), job=jcfg), git_commit=C.git_commit(), hardware=C.hardware(),
                dataset_summary=summary, wall_seconds=round(time.time() - t0, 1), peak_ram_gb=C.peak_ram_gb(), peak_vram_gb=round(vt.peak, 2),
                result=res)
    PARTS.mkdir(parents=True, exist_ok=True)
    p = Path(a.out) if a.out else PARTS / f"K{a.K}_{a.task.replace('/', '_').replace(' ', '_').replace('+', 'p')}{'_measure' if a.measure_only else ''}.json"
    p.write_text(json.dumps(C.clean(body), separators=(",", ":")))
    return body


def merge(partial=True):
    bodies = [json.loads(p.read_text()) for p in sorted(PARTS.glob("K*.json")) if "_measure" not in p.name]
    tasks = [b["result"] for b in bodies]
    for b, r in zip(bodies, tasks):
        if "methods" in r:
            C.finalise_task(r)
            r["best_one_pass"] = C.best_one_pass(r)
        r.update(wall_seconds=b["wall_seconds"], peak_ram_gb=b["peak_ram_gb"], peak_vram_gb=b["peak_vram_gb"], job_partial=b["partial"])
    out = dict(schema_version=1, experiment="e13_stage3", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               partial=bool(partial or any(b["partial"] for b in bodies)), config=dict(repo=C.load_config(), jobs=[b["config"]["job"] for b in bodies]),
               git_commit=C.git_commit(), hardware=C.hardware(), peak_ram_gb_max=max((b["peak_ram_gb"] for b in bodies), default=0),
               peak_vram_gb_max=max((b["peak_vram_gb"] for b in bodies), default=0), wall_seconds_sum=sum(b["wall_seconds"] for b in bodies),
               tasks=tasks)
    (C.E13 / "results" / "e13_stage3.json").write_text(json.dumps(C.clean(out), separators=(",", ":")))
    print(f"merged {len(tasks)} stage-3 jobs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--K", required=True)
    j.add_argument("--task", required=True)
    j.add_argument("--seeds", type=int, default=3)
    j.add_argument("--gate", action="store_true")
    j.add_argument("--hl", action="store_true")
    j.add_argument("--no-logreg", action="store_true")
    j.add_argument("--logreg-iter", type=int, default=200)
    j.add_argument("--measure-only", action="store_true")
    j.add_argument("--out", default="")
    j.add_argument("--dataset", default="pbmc68k")
    j.add_argument("--split", type=int, default=0)
    j.add_argument("--rec-cap", type=float, default=3e10)
    j.add_argument("--minf-seeds", type=int, default=1)
    j.add_argument("--hl-seeds", type=int, default=1)
    m = sub.add_parser("merge")
    m.add_argument("--final", action="store_true")
    a = ap.parse_args()
    job(a) if a.cmd == "job" else merge(partial=not a.final)
