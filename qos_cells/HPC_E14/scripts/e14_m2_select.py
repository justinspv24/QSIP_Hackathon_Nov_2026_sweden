"""E14 method 2 (and 4): supervised streaming gene selection + exact ridge. 2 passes (flagged).

Pass 1 (one pass over the training rows): count-sketch (r rows x w buckets, median decode; seeded hashes) of the
per-class sums  sum_{y=c} x  (score 'mean': |Delta| = |mean_1 - mean_0|, the E13 cs_select statistic, 2 sketches)
or also of  sum_{y=c} x^2  (score 't': |Delta| / sqrt(v1/n1 + v0/n0), 4 sketches).  Counters C = n_sketches * r * w.
After the pass: query every gene id against the sketch, keep a top-k heap (k ids + k scores).
Pass 2: exact balanced-weight ridge on the k kept genes (sufficient statistics: weighted Gram k(k+1)/2, weighted sums
2k, ids k), alpha in ALPHAS. Method 4 (recursive least squares on the same k genes, 2-pass) returns the identical
solution (RLS with P0 = I/alpha is exact ridge), with state P k(k+1)/2 + w k + ids k: same accuracies, same (b) up to
O(k); see e14_m4_rls_check.py.
Count-sketch is linear, so sketching the exact class sums == streaming the rows into the sketch (bit-for-bit up to
float summation order). The per-subset class sums are computed on the training rows of that subset only.

Memory ledgers per config (C, k):
  (a) paper accounting: counters + top-k ids + values = C + 2k  (second-moment matrix not counted, as the paper counts
      exact ridge on S hashed features as S)
  (b) true peak floats: max(pass 1: C + 2k + 3,  pass 2: k(k+1)/2 + 3k + 4)
Reference (not memory limited): 'exact' selection with all-D class statistics (oracle ranking), reported vs k only.

    python e14_m2_select.py run --gpus 8
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e14_core as K  # noqa: E402

C_GRID = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
K_GRID = [1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 44, 64, 90, 128, 181, 256, 362, 512, 724, 1024, 1448, 2048]
ROWS = [1, 3]
SCORES = ["mean", "t"]
N_SEEDS = 3
SMAX = 8192
OUT = K.E14 / "parts" / "m2"


def rankings_spec():
    spec = []
    for si in range(N_SEEDS):
        for sc in SCORES:
            for r in ROWS:
                nsk = 2 if sc == "mean" else 4
                for C in C_GRID:
                    w = C // (nsk * r)
                    if w < 1:
                        continue
                    Cr = w * nsk * r
                    kmax = min(K_GRID[-1], (SMAX - Cr) // 2)
                    if kmax < 1:
                        continue
                    spec.append(dict(seed=si, score=sc, rows=r, width=w, C=Cr, kmax=kmax))
    for sc in SCORES:
        spec.append(dict(seed=-1, score=sc, rows=0, width=0, C=-1, kmax=K_GRID[-1]))   # exact oracle ranking
    return spec


def mem(C, k):
    return C + 2 * k, max(C + 2 * k + 3, k * (k + 1) // 2 + 3 * k + 4)


def worker(gpu, q, X, y, up):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import cupy as cp
    import cupyx.scipy.sparse as csp
    spec = rankings_spec()
    while True:
        pi = q.get()
        if pi is None:
            return
        fn = OUT / f"p{pi:02d}.npz"
        if fn.exists():
            continue
        t0 = time.time()
        Xp, yp = K.pair_data(X, y, up, pi)
        D = Xp.shape[1]
        b = K.blocks(yp)
        subs = K.subsets(4)
        trm, evm = K.subset_masks(b, subs)
        Xg = csp.csr_matrix(Xp)
        X2g = csp.csr_matrix(Xp.multiply(Xp).tocsr())
        Xd = cp.asarray(Xp.toarray())
        yg = cp.asarray(yp)
        # hashes: per (seed, row) bucket index for each width + sign
        hs = {}
        for si in range(N_SEEDS):
            rng = np.random.default_rng([2024, pi, si])
            for r in range(3):
                hs[(si, r)] = (rng.integers(0, 2**31 - 1, size=D), cp.asarray(rng.choice([-1.0, 1.0], size=D)))
        acc = np.full((len(spec), len(subs), len(K_GRID), len(K.ALPHAS)), np.nan, dtype=np.float32)
        bacc = np.full_like(acc, np.nan)
        for u in range(len(subs)):
            tr, ev = cp.asarray(trm[u]), cp.asarray(evm[u])
            m1 = (tr & (yg == 1)).astype(cp.float32)
            m0 = (tr & (yg == 0)).astype(cp.float32)
            n1, n0 = float(m1.sum()), float(m0.sum())
            stats = dict(s1=Xg.T @ m1, s0=Xg.T @ m0, q1=X2g.T @ m1, q0=X2g.T @ m0)
            stats = {k_: v.astype(cp.float64) for k_, v in stats.items()}
            yev = yp[evm[u]]
            for ri, sp_ in enumerate(spec):
                if sp_["seed"] < 0:
                    est = dict(stats)
                else:
                    est = {}
                    for key in (() if sp_["score"] == "mean" else ("s1", "s0", "q1", "q0")):
                        rows = []
                        for r in range(sp_["rows"]):
                            hraw, sg = hs[(sp_["seed"], r)]
                            h = cp.asarray(hraw % sp_["width"])
                            cnt = cp.bincount(h, weights=sg * stats[key], minlength=sp_["width"])
                            rows.append(sg * cnt[h])
                        est[key] = cp.median(cp.stack(rows), axis=0) if len(rows) > 1 else rows[0]
                    if sp_["score"] == "mean":   # median of the Delta sketch rows (linear combination per row first)
                        rows = []
                        for r in range(sp_["rows"]):
                            hraw, sg = hs[(sp_["seed"], r)]
                            h = cp.asarray(hraw % sp_["width"])
                            v = stats["s1"] / n1 - stats["s0"] / n0
                            cnt = cp.bincount(h, weights=sg * v, minlength=sp_["width"])
                            rows.append(sg * cnt[h])
                        delta = cp.median(cp.stack(rows), axis=0) if len(rows) > 1 else rows[0]
                if sp_["score"] == "mean":
                    score = cp.abs(delta) if sp_["seed"] >= 0 else cp.abs(est["s1"] / n1 - est["s0"] / n0)
                else:
                    mu1, mu0 = est["s1"] / n1, est["s0"] / n0
                    v1 = cp.maximum(est["q1"] / n1 - mu1 ** 2, 1e-6)
                    v0 = cp.maximum(est["q0"] / n0 - mu0 ** 2, 1e-6)
                    score = cp.abs(mu1 - mu0) / cp.sqrt(v1 / n1 + v0 / n0)
                kmax = sp_["kmax"]
                idx = np.argsort(-cp.asnumpy(score), kind="stable")[:kmax]
                F = Xd[:, cp.asarray(idx)]
                kl = [k for k in K_GRID if k <= kmax]
                a_, b_ = K.ridge_prefix_fast(cp, F, yg, tr, ev, kl)
                acc[ri, u, :len(kl), :] = a_.T
                bacc[ri, u, :len(kl), :] = b_.T
        np.savez(fn, acc=acc, bacc=bacc, subs=np.array(subs), seconds=time.time() - t0, gpu=gpu)
        K.log(f"m2 gpu{gpu}: pair {pi} n={len(yp)} done {time.time() - t0:.1f}s")


def run(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, names, up, mult = K.load_all()
    sizes = [int(((y == c1) | (y == c2)).sum()) for c1, c2 in up]
    pis = list(range(len(up))) if a.pairs is None else [int(v) for v in a.pairs.split(",")]
    pis.sort(key=lambda p: -sizes[p])
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    for p in pis:
        q.put(p)
    for _ in range(a.gpus):
        q.put(None)
    K.log(f"m2: {len(pis)} pairs on {a.gpus} GPUs, {len(rankings_spec())} rankings x 25 subsets per pair")
    ps = [ctx.Process(target=worker, args=(g, q, X, y, up)) for g in range(a.gpus)]
    [p.start() for p in ps]
    [p.join() for p in ps]
    K.log(f"m2 done {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--pairs", default=None)
    run(ap.parse_args())
