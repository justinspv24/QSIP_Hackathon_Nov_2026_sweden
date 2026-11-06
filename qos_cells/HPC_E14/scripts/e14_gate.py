"""E14 gate: reproduce E13 Stage-4 numbers before any new method.
 (1) full ridge, all 46 pairs, E13's own ridge_cv (sklearn)   -> must equal 0.90146 (weighted CV accuracy)
 (2) signed hashing S=64, paper seed 0, all 46 pairs: E14 exact ridge engine (alpha=200, our block folds) vs E13 raw
     per-fold accuracies -> must agree (same folds, same sketch, exact solver vs sklearn's).
"""
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e14_core as K  # noqa: E402

G = {}


def t_full(pi):
    Xp, yp = K.pair_data(G["X"], G["y"], G["up"], pi)
    t0 = time.time()
    r = K.S4.ridge_cv(Xp, yp)
    return pi, r, time.time() - t0


def t_jl(pi):
    Xp, yp = K.pair_data(G["X"], G["y"], G["up"], pi)
    t0 = time.time()
    H = K.hashed(Xp, 64, G["seeds"][0]).toarray()
    b = K.blocks(yp)
    acc = []
    for k in range(5):
        pr = K.ridge_prefix(np, H, yp, b != k, b == k, [64], alphas=[200.0])[(64, 200.0)]
        acc.append(K.acc_bacc(yp[b == k], pr)[0])
    return pi, acc, time.time() - t0


if __name__ == "__main__":
    t0 = time.time()
    K.check_sha()
    X, y, names, up, mult = K.load_all()
    G.update(X=X, y=y, up=up, seeds=K.sketch_utils.sample_sketch_seeds(5))
    K.log(f"loaded D={X.shape[1]}, {len(up)} pairs, {sum(mult)} draws ({time.time()-t0:.1f}s)")
    with mp.get_context("fork").Pool(46) as pool:
        full = pool.map(t_full, range(len(up)))
        jl = pool.map(t_jl, range(len(up)))
    fa = K.weighted_mean_sem({pi: np.mean(r["acc"]) for pi, r, _ in full}, mult)
    raw = json.loads((K.ROOT.parents[1] / "qos_cells_e13/qos_cells/HPC_E13/logs/parts_stage4/raw.json").read_text())["results"]
    e13 = {r["pair"]: r["acc"] for r in raw if r["kind"] == "jl" and r["param"] == 64 and r["seed_index"] == 0}
    diffs = [abs(a - b) for pi, acc, _ in jl for a, b in zip(acc, e13[pi])]
    ja = K.weighted_mean_sem({pi: np.mean(acc) for pi, acc, _ in jl}, mult)
    e13a = K.weighted_mean_sem({pi: np.mean(e13[pi]) for pi in e13}, mult)
    body = dict(schema_version=1, experiment="e14_gate", created_utc=K.dt.datetime.now(K.dt.timezone.utc).isoformat(timespec="seconds"),
                git_commit=K.git_commit(), hardware=K.hardware(), data_sha256=K.SHA256, D=X.shape[1],
                full_ridge=dict(e14=fa, e13=K.FULL_RIDGE, pass_=abs(fa["mean"] - K.FULL_RIDGE) < 1e-5),
                jl64_seed0=dict(e14_engine=ja, e13_sklearn=e13a, max_abs_fold_diff=max(diffs), n_folds_differ=int(sum(d > 0 for d in diffs)),
                                n_folds=len(diffs), pass_=abs(ja["mean"] - e13a["mean"]) < 1e-3),
                seconds_per_pair_full_ridge=dict(mean=float(np.mean([t for *_, t in full])), max=float(max(t for *_, t in full))),
                seconds_per_pair_jl64_engine=dict(mean=float(np.mean([t for *_, t in jl])), max=float(max(t for *_, t in jl))),
                wall_seconds=round(time.time() - t0, 1), peak_ram_gb=K.peak_ram_gb())
    K.write_json(K.E14 / "results" / "e14_gate.json", body)
    print(json.dumps({k: body[k] for k in ("full_ridge", "jl64_seed0", "seconds_per_pair_full_ridge", "wall_seconds")}, indent=1))
