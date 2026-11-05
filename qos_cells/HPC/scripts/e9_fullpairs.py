"""E9 — full gene-pair space on PBMC3k: S* vs N up to N = 3.4e7 (K = 8192) and 9.4e7 (all 13,714 genes).

Same task, split, gene rule, stream semantics, methods and S* rules as E7 (scripts/09_route2_classical_memory.py),
but every pair-space quantity is computed in K x K form on the GPU (HPC/scripts/pairs_gpu.py): no per-cell pair
vector is ever built (except the hashed features for hashing+logreg, built in GPU batches and discarded).

Usage (one process per K and part, each on its own GPU):
    python e9_fullpairs.py run --K 512 --seeds 5 --part main          # gate (compare with E7)
    python e9_fullpairs.py run --K 8192 --seeds 5 --part main
    python e9_fullpairs.py run --K 13714 --seeds 3 --part main --no-reservoir
    python e9_fullpairs.py run --K 8192 --part logreg                  # logreg on all N pair features
    python e9_fullpairs.py run --K 2048 --part deff                    # exact centroid + d_eff only (E7's K)
    python e9_fullpairs.py merge                                       # -> results/e9_fullpairs.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpc_common as hc  # noqa: E402

TAG = "e9"
POSITIVES = ["B cells", "Dendritic cells", "Megakaryocytes"]
M_GRID = [33, 100, 300, 900, 2700, 8100]


def e7mod():
    spec = importlib.util.spec_from_file_location("e7route2", hc.ROOT / "scripts" / "09_route2_classical_memory.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def log(msg):
    hc.log(TAG, msg)


def s_grid_for(N):
    top = min(22, int(math.floor(math.log2(N))))
    g = [2**e for e in range(3, top + 1)]
    if g[-1] != N:
        g.append(N)
    return g


def build(X, labels, pos, K, cfg):
    from qos_cells.load_data import select_genes, stratified_split
    from pairs_gpu import PairData

    y = (labels == pos).astype(int)
    tr, te = stratified_split(y, cfg["data"]["test_frac"], int(cfg["seed"]))
    g = select_genes(X[tr], y[tr], K, cfg["data"]["min_cells_per_gene"])          # TRAIN ONLY
    rows = np.r_[tr, te]
    pd = PairData(X[rows][:, g])
    return pd, y[tr], y[te], len(tr), g


def make_evaluator(pd, ytr, yte, ntr):
    from qos_cells.readout import best_threshold, evaluate

    ab_tr, ab_te = pd.abstain[:ntr], pd.abstain[ntr:]

    def ev_scores(s):
        s = np.asarray(s, dtype=np.float64)
        a, b = s[:ntr].copy(), s[ntr:].copy()
        a[ab_tr] = -np.inf
        b[ab_te] = -np.inf
        thr = best_threshold(a, ytr)
        e = evaluate(b, yte, thr)
        return dict(balanced_accuracy=e["balanced_accuracy"], auprc=e["auprc"])

    def ev(D):
        import cupy as cp
        return ev_scores(cp.asnumpy(pd.scores(D)))

    return ev, ev_scores


def run_main(args, cfg, X, labels, tracker, out_name, exp_cfg):
    import cupy as cp
    from qos_cells.stream import class_weights
    from qos_cells.sketch import rescale_safe
    from pairs_gpu import (MG, TopS, d_eff, hashed_features, ledger, logreg_dense, sk_count_sketch, sk_hashing,
                           sk_jl_sparse)
    E7 = e7mod()

    K = args.K
    tasks = []
    for ti, pos in enumerate(POSITIVES):
        t0 = time.time()
        pd, ytr, yte, ntr, g = build(X, labels, pos, K, cfg)
        N = pd.N
        S_grid = s_grid_for(N)
        tr_rows = np.arange(ntr)
        pi_hat = float(ytr.mean())
        w_cell = class_weights(pi_hat)[ytr]
        ev, ev_scores = make_evaluator(pd, ytr, yte, ntr)
        E_tr = int(pd.npairs[:ntr].sum())
        res = dict(task=f"K={K} {pos}", positive=pos, K=K, N=int(N), index_qubits=int(math.ceil(math.log2(N))),
                   n_train=int(ntr), n_test=int(len(yte)), n_train_pos=int(ytr.sum()), n_test_pos=int(yte.sum()),
                   pi_hat=pi_hat, base_rate_test=float(yte.mean()), abstain_rate_test=float(pd.abstain[ntr:].mean()),
                   train_density=E_tr / (ntr * N), nnz_train=E_tr, nnz_test=int(pd.npairs[ntr:].sum()),
                   genes_head=[str(s) for s in np.asarray(hc_genes)[g][:12]], S_grid=S_grid)
        # ---- exact centroid ----------------------------------------------------------------------------------
        delta = pd.centroid_delta(tr_rows, ytr)
        e = ev(delta)
        res["exact_centroid"] = dict(**e, **ledger(floats=N))
        res["d_eff"] = d_eff(delta)
        res["delta_norm2"] = float(cp.sum(delta**2))
        exact_ba = e["balanced_accuracy"]
        log(f"{res['task']}: N={N:,} exact bal.acc {exact_ba:.4f} auprc {e['auprc']:.4f} d_eff {res['d_eff']:.1f} "
            f"(build {time.time() - t0:.0f} s, train nnz {E_tr:,})")
        if args.gate_only:
            tasks.append(res)
            continue
        # ---- M selection (convergence check, stop one grid step after convergence) --------------------------------
        msel, phis, run_row = [], None, None
        conv_at = None
        for mi, mo in enumerate(M_GRID[: args.max_m_steps]):
            M = int(mo * N)
            cur, evs, qs = [], [], []
            for s in range(args.seeds):
                phi, ppos, R = pd.stream(tr_rows, w_cell, M, seed=hash((K, ti, s, mo)) % (2**31))
                cur.append((phi, ppos, R))
                evs.append(ev(phi))
                ps, c = rescale_safe(cp.asnumpy(phi))
                qs.append(ev(cp.sin(cp.asarray(ps))))
            rel = [float(cp.linalg.norm(c[0] - delta) / cp.linalg.norm(delta)) for c in cur]
            row = dict(M_over_N=mo, M=M, nonzero_records_mean=float(np.mean([c[2] for c in cur])),
                       balanced_accuracy_mean=float(np.mean([x["balanced_accuracy"] for x in evs])),
                       balanced_accuracy_std=float(np.std([x["balanced_accuracy"] for x in evs])),
                       auprc_mean=float(np.mean([x["auprc"] for x in evs])), auprc_std=float(np.std([x["auprc"] for x in evs])),
                       balanced_accuracy_seeds=[round(x["balanced_accuracy"], 4) for x in evs], rel_error_mean=float(np.mean(rel)),
                       rel_error_predicted=float(np.sqrt(N / (M * ntr) * np.sum(w_cell**2 * pd.q[:ntr])) / np.sqrt(res["delta_norm2"])),
                       quantum_sin_readout_balanced_accuracy_mean=float(np.mean([x["balanced_accuracy"] for x in qs])),
                       quantum_sin_readout_auprc_mean=float(np.mean([x["auprc"] for x in qs])))
            msel.append(row)
            phis = cur
            log(f"{res['task']}: M/N={mo} stream_full bal.acc {row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f} "
                f"rel.err {row['rel_error_mean']:.3f} (pred {row['rel_error_predicted']:.3f}) records {row['nonzero_records_mean']:.2e}")
            if conv_at is not None:
                break
            if row["balanced_accuracy_mean"] >= exact_ba - 0.01:
                conv_at = mo
                if mi + 1 >= min(len(M_GRID), args.max_m_steps):
                    break
        run_row = msel[-1]
        res["M_selection"] = msel
        res["M_converged_over_N"] = conv_at
        res["M_run"], res["M_run_over_N"] = run_row["M"], run_row["M_over_N"]
        res["M_rule"] = "E7 rule: first M/N on the grid with stream_full >= exact - 0.01, run one grid step beyond (or the largest M tried)"
        res["stream_full"] = dict(balanced_accuracy=run_row["balanced_accuracy_mean"], auprc=run_row["auprc_mean"],
                                  balanced_accuracy_std=run_row["balanced_accuracy_std"], **ledger(floats=N),
                                  nonzero_records_mean=run_row["nonzero_records_mean"])
        res["quantum_sim"] = dict(index_qubits=res["index_qubits"], logical_qubits=2 * res["index_qubits"],
                                  balanced_accuracy=run_row["quantum_sin_readout_balanced_accuracy_mean"],
                                  auprc=run_row["quantum_sin_readout_auprc_mean"], M=run_row["M"],
                                  note="classical simulation of the oracle sketch (phases phi, readout p.sin(c phi)); NOT run on hardware")
        # ---- memory-limited methods ---------------------------------------------------------------------------
        jidx = cp.arange(N, dtype=cp.uint64)
        specs = [("count_sketch", "count-sketch (3 rows, median)"), ("hashing", "feature hashing (signed, 1 row)"),
                 ("jl_sparse", "sparse JL projection (4 rows)"), ("heavy_hitters_mg", "heavy hitters: Misra–Gries (S/2 + S/2)"),
                 ("heavy_hitters_top", "heavy hitters: oracle top-S |Δ|")]
        if not args.no_reservoir:
            specs.append(("reservoir", "reservoir: S nonzero records"))
        specs.append(("hashing_logreg", "feature hashing + logistic regression"))
        methods = {}
        scale = E_tr / max(1, int((~pd.abstain[:ntr]).sum()))          # = 1 / mean nonzero p (E7's logreg scale)
        for mk, (key, label) in enumerate(specs):
            t = time.time()
            per_seed = []
            n_s = args.seeds if key != "hashing_logreg" else min(args.seeds, args.hl_seeds)
            for s in range(n_s):
                phi, ppos, _ = phis[s]
                rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), K, 11, ti, s, mk]))
                seeds = [int(v) for v in rng.integers(0, 2**62, size=16)]
                metrics = {}
                if key == "hashing_logreg":
                    grid = [S for S in S_grid if S <= 2**16]
                    H = hashed_features(pd, np.arange(pd.n), grid, seeds)
                    for S in grid:
                        w, b, nit, convd = logreg_dense(H[S][:ntr], ytr, scale, max_iter=args.logreg_iter)
                        z = scale * (H[S] @ cp.asarray(w, dtype=cp.float32)).astype(cp.float64) + b
                        e = ev_scores(cp.asnumpy(z))
                        metrics[S] = dict(**e, mem=ledger(floats=S + 1, seeds=2), n_iter=nit)
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
                        d, _, _ = pd.stream(tr_rows, w_cell, None, seed=seeds[0] % (2**31) + S, n_records_override=S)
                        m = ledger(floats=S, ints=S)
                    e = ev(d)
                    metrics[S] = dict(**e, mem=m)
                    del d
                if key == "heavy_hitters_mg":
                    del mg
                if key == "heavy_hitters_top":
                    del top
                per_seed.append(metrics)
                cp.get_default_memory_pool().free_all_blocks()
            rows = E7.summarise(per_seed)
            Sst, Bst = E7.s_star(rows, exact_ba)
            methods[key] = dict(label=label, rows=rows, seeds=n_s, S_star=Sst, S_star_over_N=(Sst / N if Sst else None),
                                bytes_at_S_star=Bst, seconds=round(time.time() - t, 1))
            log(f"{res['task']}: {key:<18} S* = {Sst}  [{methods[key]['seconds']} s, {n_s} seeds]")
            res["methods"] = methods
            res["seconds"] = round(time.time() - t0, 1)
            hc.write_json(out_name, exp_cfg, dict(tasks=tasks + [res]), tracker, partial=True)
        del phis, jidx, delta
        res["seconds"] = round(time.time() - t0, 1)
        tasks.append(res)
        del pd
        cp.get_default_memory_pool().free_all_blocks()
        hc.write_json(out_name, exp_cfg, dict(tasks=tasks), tracker, partial=(ti < len(POSITIVES) - 1))
    return tasks


def run_logreg(args, cfg, X, labels, tracker, out_name, exp_cfg):
    import cupy as cp
    from pairs_gpu import ledger, logreg_pairs

    tasks = []
    for ti, pos in enumerate(POSITIVES):
        t0 = time.time()
        pd, ytr, yte, ntr, g = build(X, labels, pos, args.K, cfg)
        ev, ev_scores = make_evaluator(pd, ytr, yte, ntr)
        E_tr = int(pd.npairs[:ntr].sum())
        scale = E_tr / max(1, int((~pd.abstain[:ntr]).sum()))
        w, b, nit, convd = logreg_pairs(pd, np.arange(ntr), ytr, scale, max_iter=args.logreg_iter)
        z = scale * pd.scores(cp.asarray(w)) + b
        e = ev_scores(cp.asnumpy(z))
        r = dict(task=f"K={args.K} {pos}", positive=pos, K=args.K, N=pd.N,
                 logreg_full=dict(**e, **ledger(floats=pd.N + 1), feature_scale=scale, n_iter=nit, converged=convd,
                                  max_iter=args.logreg_iter, seconds=round(time.time() - t0, 1),
                                  method="class-balanced L2 logistic regression (C=1) on all N pair features, solved in K x K form on GPU (L-BFGS)"))
        log(f"{r['task']}: logreg_full bal.acc {e['balanced_accuracy']:.4f} auprc {e['auprc']:.4f} iters {nit} [{time.time() - t0:.0f} s]")
        tasks.append(r)
        del pd
        cp.get_default_memory_pool().free_all_blocks()
        hc.write_json(out_name, exp_cfg, dict(tasks=tasks), tracker, partial=(ti < len(POSITIVES) - 1))


def run(args):
    global hc_genes
    from qos_cells.load_data import load_pbmc3k

    cfg = hc.load_config()
    tracker = hc.Tracker()
    X, genes, labels, summary = load_pbmc3k()
    hc_genes = genes
    if args.K > X.shape[1]:
        raise SystemExit(f"K={args.K} > {X.shape[1]} genes")
    exp_cfg = dict(K=args.K, part=args.part, seeds=args.seeds, hashing_logreg_seeds=min(args.seeds, args.hl_seeds),
                   M_over_N_grid=M_GRID[: args.max_m_steps], reservoir=not args.no_reservoir, logreg_max_iter=args.logreg_iter,
                   positives=POSITIVES, split_seed=int(cfg["seed"]), test_frac=cfg["data"]["test_frac"],
                   S_grid="powers of 2 from 8 to min(2^22, N), plus N; hashing+logreg S <= 2^16",
                   stream="E7 Poissonised uniform stream, simulated by Poisson splitting (total records ~ Poisson(lambda E), uniform nonzero entry)",
                   dataset=summary)
    out_name = f"e9_part_K{args.K}_{args.part}.json"
    if args.part == "main":
        run_main(args, cfg, X, labels, tracker, out_name, exp_cfg)
    elif args.part == "logreg":
        run_logreg(args, cfg, X, labels, tracker, out_name, exp_cfg)
    elif args.part == "deff":
        args.gate_only = True
        tasks = run_main(args, cfg, X, labels, tracker, out_name, exp_cfg)
        hc.write_json(out_name, exp_cfg, dict(tasks=tasks), tracker, partial=False)
    log(f"done: {tracker.stats()}")


# ===================================================================================================================
# merge: parts -> results/e9_fullpairs.json (E7's S* rules via its _finalise / _verdict)
# ===================================================================================================================
def merge(args):
    E7 = e7mod()
    parts = sorted(hc.RESULTS.glob("e9_part_K*_*.json"))
    by_K = {}
    meta = {}
    for p in parts:
        d = json.loads(p.read_text())
        K = int(d["config"]["experiment"]["K"])
        part = d["config"]["experiment"]["part"]
        meta[p.name] = dict(partial=d["partial"], resources=d["resources"], git_commit=d["git_commit"], config=d["config"]["experiment"])
        blk = by_K.setdefault(K, dict(K=K, real=[], logreg=[], deff=[]))
        if part == "main":
            blk["real"] = d["tasks"]
        elif part == "logreg":
            blk["logreg"] = d["tasks"]
        elif part == "deff":
            blk["deff"] = d["tasks"]
    blocks = []
    for K in sorted(by_K):
        blk = by_K[K]
        lr = {t["positive"]: t["logreg_full"] for t in blk.pop("logreg")}
        for t in blk["real"]:
            if t["positive"] in lr:
                t["logreg_full"] = lr[t["positive"]]
        full = [t for t in blk["real"] if "methods" in t and "stream_full" in t]
        if blk["real"]:
            N = blk["real"][0]["N"]
        else:
            N = blk["deff"][0]["N"]
        blk.update(N=N, quantum=dict(index_qubits=int(math.ceil(math.log2(N)))))
        if full:
            sub = dict(K=K, real=full, synthetic=None, quantum=blk["quantum"])
            E7._finalise([sub])
            blk["verdict"] = E7._verdict([sub])[str(K)]
        blocks.append(blk)
    # E7 reference (K = 512, 2048) for the S*-vs-N figure
    e7 = json.loads((hc.ROOT / "results" / "e7_route2.json").read_text())
    e7_pts = []
    for b in e7["by_K"]:
        for t in b["real"]:
            best = e7["verdict"][str(b["K"])]["best_method_per_class"][t["positive"]]
            e7_pts.append(dict(source="E7", K=b["K"], N=b["N"], positive=t["positive"], exact_balanced_accuracy=t["exact_centroid"]["balanced_accuracy"],
                               best_method=best["method"], best_S_star_over_N=best["S_star_over_N"],
                               best_S_star=(best["S_star_over_N"] * b["N"] if best["S_star_over_N"] else None),
                               S_star_cell_stable={k: m.get("S_star_cell_stable") for k, m in t["methods"].items()}))
    e9_pts, deff_points, gate = [], [], []
    for blk in blocks:
        for t in blk["real"]:
            if "methods" not in t:
                continue
            best = blk["verdict"]["best_method_per_class"][t["positive"]]
            e9_pts.append(dict(source="E9", K=blk["K"], N=blk["N"], positive=t["positive"], exact_balanced_accuracy=t["exact_centroid"]["balanced_accuracy"],
                               best_method=best["method"], best_S_star_over_N=best["S_star_over_N"],
                               best_S_star=(best["S_star_over_N"] * blk["N"] if best["S_star_over_N"] else None),
                               S_star_cell_stable={k: m.get("S_star_cell_stable") for k, m in t["methods"].items()}))
        for t in blk["real"] + blk["deff"]:
            if t.get("d_eff") is None:
                continue
            bst = None
            for p in e9_pts + e7_pts:
                if p["K"] == blk["K"] and p["positive"] == t["positive"]:
                    bst = p["best_S_star_over_N"]
                    if p["source"] == "E9":
                        break
            deff_points.append(dict(label=f"PBMC3k {t['positive']} K={blk['K']}", dataset="PBMC3k (E7/E9)", task=t["positive"], K=blk["K"],
                                    N=blk["N"], d_eff=t["d_eff"], best_S_star_over_N=bst))
    # gate: K=512 rerun vs E7
    g512 = next((b for b in blocks if b["K"] == 512), None)
    e7_512 = next(b for b in e7["by_K"] if b["K"] == 512)
    if g512:
        for t in g512["real"]:
            o = next(x for x in e7_512["real"] if x["positive"] == t["positive"])
            row = dict(positive=t["positive"], exact_ba_new=t["exact_centroid"]["balanced_accuracy"], exact_ba_e7=o["exact_centroid"]["balanced_accuracy"],
                       exact_auprc_new=t["exact_centroid"]["auprc"], exact_auprc_e7=o["exact_centroid"]["auprc"])
            row["exact_match"] = abs(row["exact_ba_new"] - row["exact_ba_e7"]) < 1e-4 and abs(row["exact_auprc_new"] - row["exact_auprc_e7"]) < 1e-3
            if "methods" in t:
                row["stream_full_new"] = t["stream_full"]["balanced_accuracy"]
                row["stream_full_e7"] = o["stream_full"]["balanced_accuracy"]
                row["S_star_cell_stable"] = {k: dict(new=m.get("S_star_cell_stable"), e7=o["methods"].get(k, {}).get("S_star_cell_stable"))
                                             for k, m in t["methods"].items()}
                ratios = []
                for k, v in row["S_star_cell_stable"].items():
                    if v["new"] and v["e7"]:
                        ratios.append(abs(math.log2(v["new"] / v["e7"])))
                row["max_grid_steps_apart"] = max(ratios) if ratios else None
                row["best_new"] = g512["verdict"]["best_method_per_class"][t["positive"]]
                row["best_e7"] = e7["verdict"]["512"]["best_method_per_class"][t["positive"]]
            if "logreg_full" in t:
                row["logreg_new"] = t["logreg_full"]["balanced_accuracy"]
                row["logreg_e7"] = o.get("logreg_full", {}).get("balanced_accuracy")
            gate.append(row)
    payload = dict(by_K=blocks, e7_points=e7_pts, e9_points=e9_pts, deff_points=deff_points, gate_K512=gate, parts=meta,
                   limits=["Louvain labels are circular (clustering of the same data)",
                           "tiny rare test sets (DC 11, Mk 5 cells); one split",
                           "stream is Poissonised (Poisson splitting), not record-by-record",
                           "quantum column = classical simulation of the sketch; nothing run on hardware",
                           "S counts counters; a qubit and a float are not the same unit"])
    tracker = hc.Tracker()
    partial = any(m["partial"] for m in meta.values())
    exp_cfg = dict(merged_from=sorted(meta), note="see parts[*].config for each run's settings")
    hc.write_json("e9_fullpairs.json", exp_cfg, payload, tracker, partial=partial)
    log(f"merged {len(parts)} parts -> results/e9_fullpairs.json (partial={partial})")


hc_genes = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--K", type=int, required=True)
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--hl-seeds", type=int, default=3)
    r.add_argument("--part", choices=["main", "logreg", "deff"], default="main")
    r.add_argument("--no-reservoir", action="store_true")
    r.add_argument("--gate-only", action="store_true")
    r.add_argument("--max-m-steps", type=int, default=6)
    r.add_argument("--logreg-iter", type=int, default=300)
    sub.add_parser("merge")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else merge(a)
