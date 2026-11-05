"""E13 Stage 4 — the paper's own PBMC68k protocol (Zhao et al. 2026, Fig. 2b; code: quantum-oracle-sketching/real_datasets/
pbmc68k_svm.py), with fair classical baselines. Accuracy here is the paper's metric (5-fold CV accuracy), clearly labelled.

Reproduced from the paper's code (read-only, imported or copied verbatim where noted):
  data      scvelo PBMC68k h5ad (figshare 27686886, the file scv.datasets.pbmc68k() downloads), X = log1p(raw counts),
            genes kept if nonzero in >= 1 cell (min_samples = 1), labels obs['celltype'] (Zheng et al.), LabelEncoder order
  pairs     get_random_pairs(11 classes, 100 pairs, np.random.default_rng(42)): 100 draws WITH replacement from the 55
            pairs. Each unique pair is evaluated once; curves are averaged with the paper's multiplicities (and unweighted).
  model     RidgeClassifier(alpha=200, class_weight='balanced', random_state=42); cross_val_score(cv=5) =
            StratifiedKFold(5), no shuffle
  sketches  sketch_utils.sketch_features(X, S, mode, seed) with mode 'bucket' (balanced UNSIGNED feature hashing),
            'jl' (balanced signed sparse JL, s=1), 'jl4' (s=4); seeds = sketch_utils.sample_sketch_seeds(5) (the paper's 5 seeds)
  truncation  keep genes detected in >= min_samples cells (the paper's main sweep; machine size = genes kept)
  quantum   machine size = 2 ceil(log2(n + 2D)) + ceil(log2(s + 1)) + 4 (sweep_utils.qos_machine_size, 'svm'),
            plus our log2 D index-qubit count. Classical simulation / formula only; nothing quantum is run.
Added here (E13):
  cs_centroid  one-pass count-sketch (3 rows, median decode, S counters in total; E7's sk_count_sketch) of the class-mean
               difference on the training folds, used as a linear score; threshold maximising train balanced accuracy.
               This is the classical counterpart of the oracle-sketch classifier (same statistic, S classical counters).
  exact_centroid  the same with all D counters.
Metrics per fold: accuracy (paper) and balanced accuracy.

    python e13_stage4.py run --procs 80
    python e13_stage4.py merge
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e13_core as C  # noqa: E402

PAPER = C.ROOT.parent.parent / "QSIP_Hackathon_Nov_2026_sweden" / "quantum-oracle-sketching"
sys.path.insert(0, str(PAPER / "real_datasets"))
import sketch_utils  # noqa: E402  (the paper's code, unchanged)

DATA = C.E13 / "data" / "scvelo" / "pbmc68k_scvelo.h5ad"
URL = "https://ndownloader.figshare.com/files/27686886"
OUT = C.E13 / "logs" / "parts_stage4"
S_GRID = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
MODES = ["bucket", "jl", "jl4"]
MIN_SAMPLES = [int(v) for v in np.unique(np.logspace(np.log10(1), np.log10(20000), 30).astype(int))]   # paper: get_min_samples_sweep()
N_SEEDS = 5
G = {}


def load():
    import anndata as ad
    from sklearn.preprocessing import LabelEncoder
    a = ad.read_h5ad(DATA)
    X = sp.csr_matrix(a.X, dtype=np.float32)
    X.data = np.log1p(X.data)
    labels = np.asarray(a.obs["celltype"].astype(str))
    le = LabelEncoder()
    y = le.fit_transform(labels)
    cnt = np.asarray((X != 0).sum(0)).ravel()
    keep = np.where(cnt >= 1)[0]
    return X[:, keep].tocsr(), y, list(le.classes_), cnt[keep]


def paper_pairs(n_classes, n_pairs=100):
    rng = np.random.default_rng(42)
    all_pairs = list(combinations(range(n_classes), 2))
    idx = rng.choice(len(all_pairs), size=n_pairs, replace=True)
    return [all_pairs[i] for i in idx]


def qos_machine_size(n, D, s):
    return float(2 * np.ceil(np.log2(n + 2 * D)) + np.ceil(np.log2(s + 1)) + 4)


def ridge_cv(Xp, yp):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import cross_validate
    clf = RidgeClassifier(random_state=42, alpha=200.0, solver="auto", class_weight="balanced")
    r = cross_validate(clf, Xp, yp, cv=5, scoring=["accuracy", "balanced_accuracy"])
    return dict(acc=[float(v) for v in r["test_accuracy"]], bacc=[float(v) for v in r["test_balanced_accuracy"]])


def centroid_cv(Xp, yp, S_list, seeds_list):
    """exact centroid (S = D) and count-sketch centroid at every S in S_list, per seed; same 5 folds as cross_val_score."""
    from sklearn.model_selection import StratifiedKFold
    from qos_cells.readout import best_threshold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    D = Xp.shape[1]
    out = {"exact": dict(acc=[], bacc=[])}
    for S in S_list:
        for si in range(len(seeds_list)):
            out[(S, si)] = dict(acc=[], bacc=[])
    for tr, te in StratifiedKFold(5).split(np.zeros(len(yp)), yp):
        Xtr, Xte, ytr, yte = Xp[tr], Xp[te], yp[tr], yp[te]
        delta = np.asarray(Xtr[ytr == 1].mean(0)).ravel() - np.asarray(Xtr[ytr == 0].mean(0)).ravel()
        cols, keys = [delta], ["exact"]
        for S in S_list:
            for si, sd in enumerate(seeds_list):
                d, _ = C.E7.sk_count_sketch(delta, S, sd)
                cols.append(d)
                keys.append((S, si))
        Dm = np.stack(cols, 1)
        s_tr, s_te = np.asarray(Xtr @ Dm), np.asarray(Xte @ Dm)
        for c, k in enumerate(keys):
            thr = best_threshold(s_tr[:, c], ytr)
            pred = (s_te[:, c] > thr).astype(int)
            out[k]["acc"].append(float(accuracy_score(yte, pred)))
            out[k]["bacc"].append(float(balanced_accuracy_score(yte, pred)))
    return out


def task(t):
    """One unit of work: (pair index, kind, param, seed index) -> result dict. Uses the forked globals in G."""
    X, y, pairs = G["X"], G["y"], G["upairs"]
    pi, kind, param, si = t
    c1, c2 = pairs[pi]
    mask = (y == c1) | (y == c2)
    yp = (y[mask] == c2).astype(int)
    t0 = time.time()
    if kind == "full":
        Xp = X[mask]
        r = ridge_cv(Xp, yp)
        r["S"] = Xp.shape[1]
    elif kind == "trunc":
        cols = np.where(G["cnt"] >= param)[0]
        Xp = X[mask][:, cols]
        r = ridge_cv(Xp, yp) if Xp.shape[1] else dict(acc=None, bacc=None)
        r["S"] = int(len(cols))
    elif kind in MODES:
        Xs, _ = sketch_utils.sketch_features(X[mask], param, kind, seed=G["seeds"][si])
        r = ridge_cv(Xs, yp)
        r["S"] = int(param)
    elif kind == "centroid":
        Xp = X[mask]
        rng = np.random.default_rng(np.random.SeedSequence([7, 4, pi]))
        seeds_list = [rng.integers(0, 2**62, size=16) for _ in range(N_SEEDS)]
        S_list = [S for S in S_GRID if S < Xp.shape[1]]
        res = centroid_cv(Xp, yp, S_list, seeds_list)
        r = dict(exact=res["exact"], cs={f"{S}_{s}": res[(S, s)] for S in S_list for s in range(N_SEEDS)})
    r.update(pair=pi, kind=kind, param=param, seed_index=si, seconds=round(time.time() - t0, 2))
    return r


def run(a):
    import multiprocessing as mp
    t0 = time.time()
    X, y, names, cnt = load()
    pairs = paper_pairs(len(names))
    upairs = sorted(set(pairs))
    mult = {p: pairs.count(p) for p in upairs}
    G.update(X=X, y=y, upairs=upairs, cnt=cnt, seeds=sketch_utils.sample_sketch_seeds(N_SEEDS))
    D = X.shape[1]
    tasks = []
    for pi in range(len(upairs)):
        tasks.append((pi, "full", 0, 0))
        tasks.append((pi, "centroid", 0, 0))
        for m in MIN_SAMPLES[1:]:
            tasks.append((pi, "trunc", m, 0))
        for mode in MODES:
            for S in S_GRID:
                if S < D:
                    for si in range(N_SEEDS):
                        tasks.append((pi, mode, S, si))
    if a.limit:
        tasks = [t for t in tasks if t[0] < a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    C.log(f"stage 4: D = {D}, {len(upairs)} unique pairs (of 100 draws), {len(tasks)} tasks on {a.procs} procs")
    results = []
    with mp.get_context("fork").Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(task, tasks, chunksize=1)):
            results.append(r)
            if (i + 1) % 200 == 0 or i + 1 == len(tasks):
                C.log(f"stage 4: {i + 1}/{len(tasks)} tasks ({(time.time() - t0) / 60:.1f} min)")
                save(results, names, upairs, mult, X, y, D, t0, partial=(i + 1 < len(tasks)), a=a)
    C.log("stage 4 done")


def save(results, names, upairs, mult, X, y, D, t0, partial, a):
    pair_info = []
    for (c1, c2) in upairs:
        m = (y == c1) | (y == c2)
        Xp = X[m]
        s = int(np.diff(Xp.indptr).max())
        pair_info.append(dict(pair=[names[c1], names[c2]], multiplicity=mult[(c1, c2)], n_cells=int(m.sum()),
                              n_pos=int((y[m] == c2).sum()), max_sparsity=s, qos_machine_size_paper=qos_machine_size(int(m.sum()), D, s)))
    body = dict(schema_version=1, experiment="e13_stage4_raw", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                partial=partial, D=D, pairs=pair_info, results=results, wall_seconds=round(time.time() - t0, 1), peak_ram_gb=C.peak_ram_gb(),
                procs=a.procs)
    (OUT / "raw.json").write_text(json.dumps(C.clean(body), separators=(",", ":")))
    merge()


def merge():
    raw = json.loads((OUT / "raw.json").read_text())
    pairs, res, D = raw["pairs"], raw["results"], raw["D"]
    w = np.array([p["multiplicity"] for p in pairs], dtype=float)
    P = len(pairs)

    def agg(per_pair, metric):
        """per_pair: {pair index: list of fold scores (all seeds)} -> weighted mean over pairs + s.e.m."""
        idx = [i for i in range(P) if per_pair.get(i)]
        if not idx:
            return None
        v = np.array([np.mean(per_pair[i]) for i in idx])
        ww = w[idx]
        m = float(np.sum(ww * v) / ww.sum())
        sem = float(np.sqrt(np.sum(ww * (v - m) ** 2) / ww.sum()) / math.sqrt(len(idx)))
        return dict(mean=m, sem=sem, unweighted_mean=float(v.mean()), n_pairs=len(idx))

    curves, labels = {}, {}
    for kind in MODES + ["trunc"]:
        Ss = sorted({r["param"] for r in res if r["kind"] == kind})
        rows = []
        for S in Ss:
            rr = [r for r in res if r["kind"] == kind and r["param"] == S and r["acc"] is not None]
            pa, pb = {}, {}
            for r in rr:
                pa.setdefault(r["pair"], []).extend(r["acc"])
                pb.setdefault(r["pair"], []).extend(r["bacc"])
            A, B = agg(pa, "acc"), agg(pb, "bacc")
            if A:
                size = float(np.mean([r["S"] for r in rr]))
                rows.append(dict(S=size, param=S, accuracy_mean=A["mean"], accuracy_sem=A["sem"], accuracy_unweighted=A["unweighted_mean"],
                                 balanced_accuracy_mean=B["mean"], balanced_accuracy_sem=B["sem"], n_pairs=A["n_pairs"]))
        curves[kind] = sorted(rows, key=lambda r: r["S"])
    pa, pb, ca, cb = {}, {}, {}, {}
    for r in res:
        if r["kind"] == "full":
            pa[r["pair"]], pb[r["pair"]] = r["acc"], r["bacc"]
        if r["kind"] == "centroid":
            ca[r["pair"]], cb[r["pair"]] = r["exact"]["acc"], r["exact"]["bacc"]
    full_a, full_b = agg(pa, "acc"), agg(pb, "bacc")
    ex_a, ex_b = agg(ca, "acc"), agg(cb, "bacc")
    rows = []
    for S in S_GRID:
        qa, qb = {}, {}
        for r in res:
            if r["kind"] == "centroid":
                for k, v in r["cs"].items():
                    if int(k.split("_")[0]) == S:
                        qa.setdefault(r["pair"], []).extend(v["acc"])
                        qb.setdefault(r["pair"], []).extend(v["bacc"])
        A, B = agg(qa, "acc"), agg(qb, "bacc")
        if A:
            rows.append(dict(S=S, accuracy_mean=A["mean"], accuracy_sem=A["sem"], accuracy_unweighted=A["unweighted_mean"],
                             balanced_accuracy_mean=B["mean"], balanced_accuracy_sem=B["sem"], n_pairs=A["n_pairs"]))
    curves["cs_centroid"] = rows
    labels = {"bucket": "paper: feature hashing, unsigned balanced buckets + ridge", "jl": "paper: sparse JL s=1 (signed hashing) + ridge",
              "jl4": "sparse JL s=4 + ridge", "trunc": "paper: rare-gene truncation + ridge",
              "cs_centroid": "one-pass count-sketch centroid (3 rows)"}

    def s_star(rows, target, tol=0.01):
        ok = [r["accuracy_mean"] >= target - tol for r in rows]
        for i in range(len(rows)):
            if all(ok[i:]):
                return rows[i]["S"]
        return None

    sstar = {k: dict(vs_full_ridge=s_star(v, full_a["mean"]) if full_a else None,
                     vs_exact_centroid=s_star(v, ex_a["mean"]) if ex_a else None) for k, v in curves.items()}
    qs = [p["qos_machine_size_paper"] for p in pairs]
    out = dict(schema_version=1, experiment="e13_stage4", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               partial=raw["partial"],
               config=dict(repo=C.load_config(), data=dict(url=URL, sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                                                            n_cells_file=65877, n_genes_file=33939, preprocessing="log1p(raw counts), genes with >= 1 count"),
                           paper_code=dict(path="quantum-oracle-sketching/real_datasets/pbmc68k_svm.py + sketch_utils.py",
                                           commit="07e91674c167b2ba28f85c3cae2da8228e906731"),
                           classifier="RidgeClassifier(alpha=200, class_weight='balanced', random_state=42), 5-fold StratifiedKFold (no shuffle)",
                           pairs="paper's 100 draws with replacement (rng 42); unique pairs evaluated once; means weighted by multiplicity",
                           S_grid=S_GRID, modes=MODES, min_samples=MIN_SAMPLES, sketch_seeds=N_SEEDS,
                           metric="5-fold CV ACCURACY (the paper's metric); balanced accuracy stored alongside"),
               git_commit=C.git_commit(), hardware=C.hardware(), wall_seconds=raw["wall_seconds"], peak_ram_gb=raw["peak_ram_gb"],
               peak_vram_gb=0.0, procs=raw["procs"], D=D, N=D, n_pairs=P, n_draws=int(w.sum()), pairs=pairs,
               full_ridge=dict(**(full_a or {}), balanced=full_b, label="ridge on all D genes"),
               exact={"accuracy_mean": (full_a or {}).get("mean"), "label": "ridge, all genes"},
               exact_centroid=dict(accuracy=ex_a, balanced_accuracy=ex_b),
               curve_methods=["bucket", "jl", "jl4", "trunc", "cs_centroid"], labels=labels, curves=curves, S_star_tol001=sstar,
               quantum=dict(index_qubits=int(math.ceil(math.log2(D))), logical_qubits_approx=2 * int(math.ceil(math.log2(D))),
                            paper_machine_size_mean=float(np.mean(qs)), paper_machine_size_range=[float(min(qs)), float(max(qs))],
                            note="formula only (paper's accounting: qubits + 1 float); no quantum run; ideal quantum accuracy = full ridge"))
    (C.E13 / "results" / "e13_stage4.json").write_text(json.dumps(C.clean(out), separators=(",", ":")))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--procs", type=int, default=16)
    r.add_argument("--limit", type=int, default=0, help="only the first n unique pairs (timing test)")
    sub.add_parser("merge")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else merge()
