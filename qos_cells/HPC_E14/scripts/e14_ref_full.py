"""E14 reference: full ridge on ALL genes (the paper's classifier, sklearn RidgeClassifier balanced, sparse input) with
alpha chosen by the SAME nested CV as the E14 methods (alpha in ALPHAS, on the 25 subsets of e14_core.subsets).
alpha = 200 alone reproduces E13 / the paper (0.90146). Output parts/ref_full/p{pi}_a{alpha}.npz: acc, bacc per subset.

    python e14_ref_full.py --procs 100
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e14_core as K  # noqa: E402

OUT = K.E14 / "parts" / "ref_full"
ALPHAS = K.ALPHAS + [200.0]
G = {}


def task(t):
    pi, al = t
    fn = OUT / f"p{pi:02d}_a{al:g}.npz"
    if fn.exists():
        return 0.0
    from sklearn.linear_model import RidgeClassifier
    t0 = time.time()
    Xp, yp = K.pair_data(G["X"], G["y"], G["up"], pi)
    subs = K.subsets(4)
    tr, ev = K.subset_masks(K.blocks(yp), subs)
    acc, bacc = [], []
    for u in range(len(subs)):
        clf = RidgeClassifier(random_state=42, alpha=al, class_weight="balanced").fit(Xp[tr[u]], yp[tr[u]])
        a, b = K.acc_bacc(yp[ev[u]], clf.predict(Xp[ev[u]]))
        acc.append(a); bacc.append(b)
    np.savez(fn, acc=np.array(acc), bacc=np.array(bacc), subs=np.array(subs), seconds=time.time() - t0)
    return time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=100)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, y, names, up, mult = K.load_all()
    G.update(X=X, y=y, up=up)
    sizes = [int(((y == c1) | (y == c2)).sum()) for c1, c2 in up]
    tasks = sorted([(pi, al) for pi in range(len(up)) for al in ALPHAS], key=lambda t: (-sizes[t[0]], t[1]))
    K.log(f"ref_full: {len(tasks)} tasks")
    with mp.get_context("fork").Pool(a.procs) as pool:
        for i, s in enumerate(pool.imap_unordered(task, tasks)):
            if (i + 1) % 23 == 0:
                K.log(f"ref_full {i + 1}/{len(tasks)} ({(time.time() - t0) / 60:.1f} min)")
    K.log(f"ref_full done {(time.time() - t0) / 60:.1f} min")
