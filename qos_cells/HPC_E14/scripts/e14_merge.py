"""E14 merge: nested-CV selection under a memory budget, aggregation as in E13 Stage 4, JSON out.

For a method, a pair and a seed: every config c has inner-validation accuracies (outer fold k, inner block j), outer
test accuracy / balanced accuracy, and two memory numbers mem_a (paper accounting) and mem_b (true peak floats).
At budget S under accounting r: candidates = {c : mem_r(c) <= S}; per outer fold k pick argmax_c mean_j inner acc
(ties -> smaller memory); report that config's test metrics on fold k. Mean over the 5 folds, then over seeds, gives
the pair's number; pairs are averaged with the paper's draw multiplicities (E13 weighted_mean_sem).

    python e14_merge.py            -> results/e14_stage1.json (methods present in parts/)
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e14_core as K  # noqa: E402

TARGET = K.FULL_RIDGE - 0.01


def select(inner, test, testb, mem, S):
    """inner (5, 4, C), test/testb (5, C), mem (C,) -> (acc, bacc, mem index list) averaged over folds, or None."""
    cand = np.where(mem <= S)[0]
    if len(cand) == 0:
        return None
    cand = cand[np.argsort(mem[cand], kind="stable")]
    picks = []
    for k in range(5):
        sc = np.nanmean(inner[k][:, cand], axis=0)
        sc = np.where(np.isnan(sc), -1, sc)
        picks.append(cand[int(np.argmax(sc))])
    return (float(np.mean([test[k, c] for k, c in enumerate(picks)])), float(np.mean([testb[k, c] for k, c in enumerate(picks)])), picks)


def split_subsets(arr, subs):
    """arr (U, C...) -> inner (5, 4, C...), test (5, C...)"""
    subs = [tuple(s) for s in subs]
    inner = np.stack([np.stack([arr[subs.index((k, j))] for j in range(5) if j != k]) for k in range(5)])
    test = np.stack([arr[subs.index((k, -1))] for k in range(5)])
    return inner, test


def curve(per_pair_seed, mult, S_grid, crit="acc", regs=("a", "b")):
    """per_pair_seed: {pi: [ (inner_acc, inner_bacc, test, testb, mem_a, mem_b) per seed ]} -> {'a': rows, 'b': rows}
    crit: inner metric used for selection ('acc' = the paper's metric, 'bacc' = balanced accuracy)."""
    out = {}
    for reg in regs:
        rows = []
        for S in S_grid:
            pa, pb, ma, mb = {}, {}, {}, {}
            for pi, seeds in per_pair_seed.items():
                va, vb, xa, xb = [], [], [], []
                for (inner_a, inner_b, test, testb, mem_a, mem_b) in seeds:
                    r = select(inner_a if crit == "acc" else inner_b, test, testb, mem_a if reg == "a" else mem_b, S)
                    if r is None:
                        continue
                    va.append(r[0]); vb.append(r[1])
                    xa.append(np.mean(mem_a[r[2]])); xb.append(np.mean(mem_b[r[2]]))
                if va:
                    pa[pi], pb[pi], ma[pi], mb[pi] = np.mean(va), np.mean(vb), np.mean(xa), np.mean(xb)
            if len(pa) < len(per_pair_seed) or not pa:
                continue          # budget infeasible for some pair -> no point
            A, B = K.weighted_mean_sem(pa, mult), K.weighted_mean_sem(pb, mult)
            rows.append(dict(S=S, accuracy_mean=A["mean"], accuracy_sem=A["sem"], accuracy_unweighted=A["unweighted_mean"],
                             balanced_accuracy_mean=B["mean"], balanced_accuracy_sem=B["sem"], n_pairs=A["n_pairs"],
                             mem_a_chosen=K.weighted_mean_sem(ma, mult)["mean"], mem_b_chosen=K.weighted_mean_sem(mb, mult)["mean"]))
        out[reg] = rows
    return out


def s_star(rows, target=TARGET):
    ok = [r["accuracy_mean"] >= target for r in rows]
    for i in range(len(rows)):
        if all(ok[i:]):
            return rows[i]["S"]
    return None


# ---------------------------------------------------------------- method 1
def load_m1(npairs, part="m1", prefix="m1", memf=None):
    import e14_m1_sgd as M1
    memf = memf or M1.mem
    res = {}
    for loss in ("sq", "log"):
        for snap, ep in ((0, "1p"), (1, "5ep")):
            pps = {}
            for pi in range(npairs):
                seeds = []
                for si in range(M1.N_SEEDS):
                    fs = [K.E14 / "parts" / part / f"p{pi:02d}_S{S}_s{si}_{loss}.npz" for S in M1.SH_GRID]
                    if not all(f.exists() for f in fs):
                        continue
                    I, IB, T, TB, MA, MB = [], [], [], [], [], []
                    for S, f in zip(M1.SH_GRID, fs):
                        z = np.load(f)
                        o = z["out"][:, :, snap]                          # (U, G, avg, metric)
                        U, NG = o.shape[:2]
                        flat_a = o[..., 0].reshape(U, -1)                 # config = (g, avg)
                        flat_b = o[..., 1].reshape(U, -1)
                        inner, test = split_subsets(flat_a, z["subs"])
                        innerb, testb = split_subsets(flat_b, z["subs"])
                        I.append(inner); IB.append(innerb); T.append(test); TB.append(testb)
                        for g in range(NG):
                            for avg in (0, 1):
                                a, b = memf(S, avg)
                                MA.append(a); MB.append(b)
                    seeds.append((np.concatenate(I, -1), np.concatenate(IB, -1), np.concatenate(T, -1), np.concatenate(TB, -1), np.array(MA, float), np.array(MB, float)))
                if seeds:
                    pps[pi] = seeds
            res[f"{prefix}_{loss}_{ep}"] = pps
    return res


# ---------------------------------------------------------------- method 2
def load_m2(npairs):
    import e14_m2_select as M2
    spec = M2.rankings_spec()
    pps, ora = {}, {}
    for pi in range(npairs):
        f = K.E14 / "parts" / "m2" / f"p{pi:02d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        acc, bacc, subs = z["acc"], z["bacc"], z["subs"]
        seeds = []
        for si in range(M2.N_SEEDS):
            ris = [i for i, s in enumerate(spec) if s["seed"] == si]
            A = acc[ris].transpose(1, 0, 2, 3).reshape(acc.shape[1], -1)       # (U, rank*k*alpha)
            B = bacc[ris].transpose(1, 0, 2, 3).reshape(acc.shape[1], -1)
            ma, mb = [], []
            for i in ris:
                for k in M2.K_GRID:
                    for _ in K.ALPHAS:
                        a, b = M2.mem(spec[i]["C"], k) if k <= spec[i]["kmax"] else (np.inf, np.inf)
                        ma.append(a); mb.append(b)
            inner, test = split_subsets(A, subs)
            innerb, testb = split_subsets(B, subs)
            seeds.append((inner, innerb, test, testb, np.array(ma, float), np.array(mb, float)))
        pps[pi] = seeds
        # oracle (exact class statistics) ranking: curve vs k (score and alpha by nested CV)
        ris = [i for i, s in enumerate(spec) if s["seed"] < 0]
        A = acc[ris].transpose(1, 0, 2, 3).reshape(acc.shape[1], -1)
        B = bacc[ris].transpose(1, 0, 2, 3).reshape(acc.shape[1], -1)
        kk = np.array([k for _ in ris for k in M2.K_GRID for _a in K.ALPHAS], float)
        inner, test = split_subsets(A, subs)
        innerb, testb = split_subsets(B, subs)
        ora[pi] = [(inner, innerb, test, testb, kk, kk)]
    return dict(m2_select=pps), ora


def load_grid(name, npairs, memf, cfg_of):
    """m0 / m3: npz acc (seeds, cfg, U, alpha) -> per-seed tuples; memf(cfg) -> (a, b)"""
    pps = {}
    for pi in range(npairs):
        f = K.E14 / "parts" / name / f"p{pi:02d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        cfg = cfg_of(z)
        na = z["acc"].shape[-1]
        ma = np.array([memf(c)[0] for c in cfg for _ in range(na)], float)
        mb = np.array([memf(c)[1] for c in cfg for _ in range(na)], float)
        seeds = []
        for si in range(z["acc"].shape[0]):
            A = z["acc"][si].transpose(1, 0, 2).reshape(z["acc"].shape[2], -1)
            B = z["bacc"][si].transpose(1, 0, 2).reshape(z["acc"].shape[2], -1)
            inner, test = split_subsets(A, z["subs"])
            innerb, testb = split_subsets(B, z["subs"])
            seeds.append((inner, innerb, test, testb, ma, mb))
        pps[pi] = seeds
    return pps


def ref_full(npairs, mult):
    """full ridge on all genes: alpha = 200 (paper) and alpha by nested CV (acc / bacc)."""
    out = {}
    per = {}
    for pi in range(npairs):
        fs = {al: K.E14 / "parts" / "ref_full" / f"p{pi:02d}_a{al:g}.npz" for al in K.ALPHAS + [200.0]}
        if not all(f.exists() for f in fs.values()):
            continue
        zs = {al: np.load(f) for al, f in fs.items()}
        subs = zs[200.0]["subs"]
        A = np.stack([zs[al]["acc"] for al in K.ALPHAS], 1)
        B = np.stack([zs[al]["bacc"] for al in K.ALPHAS], 1)
        inner, test = split_subsets(A, subs)
        innerb, testb = split_subsets(B, subs)
        per[pi] = [(inner, innerb, test, testb, np.zeros(len(K.ALPHAS)), np.zeros(len(K.ALPHAS)))]
        _, t200 = split_subsets(zs[200.0]["acc"][:, None], subs)
        _, b200 = split_subsets(zs[200.0]["bacc"][:, None], subs)
        out.setdefault("a200", {})[pi] = (t200.mean(), b200.mean())
    if not per:
        return None
    res = {}
    for crit in ("acc", "bacc"):
        r = curve(per, mult, [0], crit=crit, regs=("a",))["a"]
        res[f"alpha_nested_{crit}"] = r[0] if r else None
    a200 = out["a200"]
    res["alpha200"] = dict(accuracy_mean=K.weighted_mean_sem({p: v[0] for p, v in a200.items()}, mult)["mean"],
                           balanced_accuracy_mean=K.weighted_mean_sem({p: v[1] for p, v in a200.items()}, mult)["mean"])
    res["n_pairs"] = len(per)
    return res


LABELS = {
    "m1_sq_1p": "hashing + SGD, squared loss (ridge), 1 pass",
    "m1_log_1p": "hashing + SGD, logistic loss, 1 pass",
    "m1_sq_5ep": "hashing + SGD, squared loss, 5 epochs (multi-pass)",
    "m1_log_5ep": "hashing + SGD, logistic loss, 5 epochs (multi-pass)",
    "m1b_sq_1p": "hashing + AdaGrad SGD (VW default), squared loss, 1 pass",
    "m1b_log_1p": "hashing + AdaGrad SGD (VW default), logistic loss, 1 pass",
    "m1b_sq_5ep": "hashing + AdaGrad SGD, squared, 5 epochs (multi-pass)",
    "m1b_log_5ep": "hashing + AdaGrad SGD, logistic, 5 epochs (multi-pass)",
    "m2_select": "count-sketch supervised gene selection + exact ridge (2 passes)",
    "m0_hashridge": "control: paper hashing (signed, 'jl') + exact ridge, alpha by the same nested CV",
    "m3_fd": "hashing to d + Frequent Directions (rank ell) ridge, 1 pass",
    "m2_oracle_k": "reference: exact |Delta| / t-score top-k genes + ridge (selection NOT memory limited), x = k",
}
PASSES = {"m0_hashridge": 1, "m3_fd": 1, "m1b_sq_1p": 1, "m1b_log_1p": 1, "m1b_sq_5ep": 5, "m1b_log_5ep": 5, "m1_sq_1p": 1, "m1_log_1p": 1, "m1_sq_5ep": 5, "m1_log_5ep": 5, "m2_select": 2, "m2_oracle_k": 2}


def main():
    t0 = time.time()
    e13 = json.loads(K.E13_JSON.read_text())
    mult = [p["multiplicity"] for p in e13["pairs"]]
    P = len(mult)
    methods = {}
    if glob.glob(str(K.E14 / "parts" / "m1" / "*.npz")):
        methods.update(load_m1(P))
    if glob.glob(str(K.E14 / "parts" / "m1b" / "*.npz")) and "m1b" not in os.environ.get("E14_SKIP", ""):
        import e14_m1b_adagrad as M1B
        methods.update(load_m1(P, "m1b", "m1b", M1B.mem))
    ora = {}
    if glob.glob(str(K.E14 / "parts" / "m2" / "*.npz")):
        m2, ora = load_m2(P)
        methods.update(m2)
    import e14_m0_hashridge as M0
    import e14_m3_fd as M3
    skip = os.environ.get("E14_SKIP", "").split(",")      # e.g. E14_SKIP=m0 while m0 is incomplete
    if "m0" not in skip:
        methods["m0_hashridge"] = load_grid("m0", P, M0.mem, lambda z: K.S_GRID)
    if "m3" not in skip:
        methods["m3_fd"] = load_grid("m3", P, lambda c: M3.mem(*c), lambda z: [tuple(c) for c in z["cfg"]])
    curves, sstar, npairs = {}, {}, {}
    for name, pps in methods.items():
        if not pps:
            continue
        curves[name] = curve(pps, mult, K.S_GRID)
        curves[name + "__tuned_on_bacc"] = curve(pps, mult, K.S_GRID, crit="bacc")
        npairs[name] = len(pps)
        sstar[name] = {reg: s_star(curves[name][reg]) for reg in ("a", "b")}
    if ora:
        import e14_m2_select as M2
        c = curve(ora, mult, M2.K_GRID, regs=("a",))
        curves["m2_oracle_k"] = dict(k=c["a"])
        npairs["m2_oracle_k"] = len(ora)
    ref = ref_full(P, mult)
    body = dict(schema_version=1, experiment="e14_stage1", created_utc=K.dt.datetime.now(K.dt.timezone.utc).isoformat(timespec="seconds"),
                partial=any(n < P for n in npairs.values()), git_commit=K.git_commit(), hardware=K.hardware(),
                config=dict(data=dict(path=str(K.DATA), sha256=K.SHA256), D=e13["D"], pairs="E13 Stage 4: 46 unique of the paper's 100 draws, weighted by multiplicity",
                            outer_cv="StratifiedKFold(5), no shuffle (= E13 / paper cross_val_score)",
                            nested_cv="inner folds = the 4 other outer blocks; hyperparameters and memory split chosen by mean inner accuracy only",
                            S_grid=K.S_GRID, target=f"full ridge {K.FULL_RIDGE} - 0.01 = {TARGET:.5f}",
                            accounting=dict(a="paper: model registers + adaptive sketch state (counters, ids, values); Gram / covariance and per-step workspace NOT counted (the paper plots exact ridge on S hashed features at machine size S)",
                                            b="true peak floats held at any time: weights, averaged copies, bias, Gram k(k+1)/2, sums, counters, ids, scalars")),
                labels=LABELS, passes=PASSES, n_pairs=npairs, curves=curves, S_star_within_001_of_full_ridge=sstar,
                full_ridge=K.FULL_RIDGE, full_ridge_reference=ref, merge_seconds=round(time.time() - t0, 1))
    K.write_json(K.E14 / "results" / "e14_stage1.json", body)
    print("full ridge reference:", ref)
    for n, c in curves.items():
        for reg, rows in c.items():
            print(n, reg, [(r["S"], round(r["accuracy_mean"], 3)) for r in rows])
    print("S* (within 0.01 of full ridge):", sstar)


if __name__ == "__main__":
    main()
