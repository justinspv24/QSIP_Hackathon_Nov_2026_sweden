"""E14 control m0: exact balanced ridge on S signed-hashed features (the paper's 'jl' sketch, paper seeds 0-2) with alpha
chosen by the SAME nested CV as the E14 methods. = E13's sketch + ridge, re-tuned like the new methods, so that the
new learners are compared with the paper's sketch at equal tuning. One pass (sufficient statistics).
  (a) = S   (paper accounting)      (b) = S(S+1)/2 + 2S + 3   (weighted Gram + weighted sums + scalars)

    python e14_m0_hashridge.py --gpus 8
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e14_core as K  # noqa: E402

OUT = K.E14 / "parts" / "m0"
N_SEEDS = 3


def mem(S):
    return S, S * (S + 1) // 2 + 2 * S + 3


def worker(gpu, q, X, y, up):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import cupy as cp
    seeds = K.sketch_utils.sample_sketch_seeds(5)
    while True:
        pi = q.get()
        if pi is None:
            return
        fn = OUT / f"p{pi:02d}.npz"
        if fn.exists():
            continue
        t0 = time.time()
        Xp, yp = K.pair_data(X, y, up, pi)
        subs = K.subsets(4)
        trm, evm = K.subset_masks(K.blocks(yp), subs)
        yg = cp.asarray(yp)
        acc = np.full((N_SEEDS, len(K.S_GRID), len(subs), len(K.ALPHAS)), np.nan, np.float32)
        bacc = np.full_like(acc, np.nan)
        for si in range(N_SEEDS):
            for j, S in enumerate(K.S_GRID):
                F = cp.asarray(K.hashed(Xp, S, seeds[si]).toarray())
                for u in range(len(subs)):
                    a_, b_ = K.ridge_prefix_fast(cp, F, yg, cp.asarray(trm[u]), cp.asarray(evm[u]), [S])
                    acc[si, j, u], bacc[si, j, u] = a_[:, 0], b_[:, 0]
                del F
        np.savez(fn, acc=acc, bacc=bacc, subs=np.array(subs), seconds=time.time() - t0)
        K.log(f"m0 gpu{gpu}: pair {pi} n={len(yp)} {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, names, up, mult = K.load_all()
    sizes = [int(((y == c1) | (y == c2)).sum()) for c1, c2 in up]
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    for p in sorted(range(len(up)), key=lambda p: -sizes[p]):
        q.put(p)
    for _ in range(a.gpus):
        q.put(None)
    ps = [ctx.Process(target=worker, args=(g, q, X, y, up)) for g in range(a.gpus)]
    [p.start() for p in ps]
    [p.join() for p in ps]
    K.log("m0 done")
