"""E14 method 1: signed feature hashing (the paper's 'jl' s=1 sketch, paper seeds) + streaming SGD (Vowpal-Wabbit style).

Learner state (what it keeps): S hashed weights (stored as scale * u, lazy L2), bias, running class counts (2), running
mean squared row norm (1), step counter (1); averaged variant additionally z (S) + bias sum (1) + scale sum (1)
(lazy Polyak averaging, Xu 2011 / Bottou: wbar_t = (Ssum_t u_t - z_t)/t).  The current sample is input, not state:
w.(Hx) and the update are computed by hashing the sparse row on the fly (simulated here with the precomputed hashed CSR).

Per sample t (only rows of the subset's training set, in a seeded random order, 1 or 5 epochs; snapshot after epoch 1
and after epoch 5):
  c     = n / (2 n_class)  (the paper's balanced weight; class counts known up front = label-only pre-count, 2 ints;
          the first run used running counts n_seen/(2 n_seen(class)), kept in parts/m1_runningw as a variant)
  eta_t = eta0 / ((qbar_t + 1) sqrt(t))                     (normalised step: qbar = running mean |x|^2; +1 = bias)
  p = w.x + b ;  g = c (p - y) [squared / ridge]  or  g = -c y sigmoid(-y p) [logistic],   y in {-1,+1}
  w <- (1 - eta lam) w - eta g x ;   b <- b - eta g
Hyperparameters (eta0, lam) and, at a given memory budget, the hash size S_h and plain/averaged weights are chosen by
nested CV (inner folds = the other outer blocks; e14_core.subsets) on the training folds only.

    python e14_m1_sgd.py run --procs 112        (tasks = pair x S_h x seed x loss)
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(__file__))
import e14_core as K  # noqa: E402

KNOWN_COUNTS = True   # balanced weights n/(2 n_c) from known class counts (label-only pre-count); False = running counts
SH_GRID = [8, 16, 22, 32, 44, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
ETAS = [0.03, 0.1, 0.3, 1.0]
LAMS = [1e-4, 1e-3, 1e-2, 1e-1]
N_SEEDS = 3
EPOCHS = 5
SNAPS = (1, 5)
OUT = K.E14 / "parts" / "m1"
G = {}


@njit(cache=True, fastmath=False)
def _eval(indptr, ind, val, y, ev, w, b, out_acc):
    n = len(y)
    tot = 0.0
    corr = 0.0
    c1 = 0.0
    t1 = 0.0
    c0 = 0.0
    t0 = 0.0
    for i in range(n):
        if not ev[i]:
            continue
        s = b
        for p in range(indptr[i], indptr[i + 1]):
            s += w[ind[p]] * val[p]
        pr = 1 if s > 0 else 0
        tot += 1
        if y[i] == 1:
            t1 += 1
            if pr == 1:
                c1 += 1
                corr += 1
        else:
            t0 += 1
            if pr == 0:
                c0 += 1
                corr += 1
    out_acc[0] = corr / tot
    if t1 > 0 and t0 > 0:
        out_acc[1] = 0.5 * (c1 / t1 + c0 / t0)
    elif t1 > 0:
        out_acc[1] = c1 / t1
    else:
        out_acc[1] = c0 / t0


@njit(cache=True, fastmath=False)
def sgd_all(indptr, ind, val, norm2, y, perms, tr, ev, S, loss, etas, lams, out):
    """out[u, g, snap, avg, metric]"""
    U = tr.shape[0]
    NG = len(etas)
    E = perms.shape[0]
    for u in range(U):
        for g in range(NG):
            eta0 = etas[g]
            lam = lams[g]
            uu = np.zeros(S)
            z = np.zeros(S)
            sc = 1.0
            ssum = 0.0
            b = 0.0
            bsum = 0.0
            t = 0
            n1 = 0
            ntot = 0
            npos = 0
            for i in range(len(y)):
                if tr[u, i]:
                    ntot += 1
                    npos += y[i]
            cpos = ntot / (2.0 * npos)
            cneg = ntot / (2.0 * (ntot - npos))
            qs = 0.0
            si = 0
            for e in range(E):
                for ii in range(perms.shape[1]):
                    i = perms[e, ii]
                    if not tr[u, i]:
                        continue
                    t += 1
                    if y[i] == 1:
                        n1 += 1
                        c = cpos if KNOWN_COUNTS else t / (2.0 * n1)
                        ys = 1.0
                    else:
                        c = cneg if KNOWN_COUNTS else t / (2.0 * (t - n1))
                        ys = -1.0
                    qs += norm2[i]
                    eta = eta0 / ((qs / t + 1.0) * np.sqrt(t))
                    p = b
                    for q in range(indptr[i], indptr[i + 1]):
                        p += sc * uu[ind[q]] * val[q]
                    if loss == 0:
                        gr = c * (p - ys)
                    else:
                        m = ys * p
                        if m > 30:
                            gr = 0.0
                        else:
                            gr = -c * ys / (1.0 + np.exp(m))
                    sc_new = sc * (1.0 - eta * lam)
                    f = -eta * gr / sc_new
                    for q in range(indptr[i], indptr[i + 1]):
                        d = f * val[q]
                        uu[ind[q]] += d
                        z[ind[q]] += ssum * d
                    ssum += sc_new
                    sc = sc_new
                    b -= eta * gr
                    bsum += b
                    if sc < 1e-6:
                        for j in range(S):
                            uu[j] *= sc
                        ssum /= sc
                        sc = 1.0
                if si < 2 and e + 1 == (1 if si == 0 else E):
                    w = sc * uu
                    _eval(indptr, ind, val, y, ev[u], w, b, out[u, g, si, 0])
                    wb = (ssum * uu - z) / t
                    _eval(indptr, ind, val, y, ev[u], wb, bsum / t, out[u, g, si, 1])
                    si += 1


def mem(S_h, avg):
    """(a) paper accounting: stored model registers (weights; + averaged copy); bias/scalars excluded as in the
    paper's adaptive survey.  (b) true peak floats: + bias, 2 class counts, mean norm, step counter (+ bias sum, scale
    sum for the averaged variant) [+ the lazy-L2 scale]."""
    a = S_h * (2 if avg else 1)
    b = a + 6 + (2 if avg else 0)
    return a, b


def task(t):
    pi, S_h, si, loss = t
    fn = OUT / f"p{pi:02d}_S{S_h}_s{si}_{'sq' if loss == 0 else 'log'}.npz"
    if fn.exists():
        return str(fn), 0.0
    t0 = time.time()
    Xp, yp = K.pair_data(G["X"], G["y"], G["up"], pi)
    H = K.hashed(Xp, S_h, G["seeds"][si])
    b = K.blocks(yp)
    subs = K.subsets(4)
    tr, ev = K.subset_masks(b, subs)
    rng = np.random.default_rng([11, pi, si])
    perms = np.stack([rng.permutation(len(yp)) for _ in range(EPOCHS)]).astype(np.int64)
    norm2 = np.asarray(H.multiply(H).sum(1)).ravel().astype(np.float64)
    gs = [(e, l) for e in ETAS for l in LAMS]
    out = np.zeros((len(subs), len(gs), 2, 2, 2))
    sgd_all(H.indptr.astype(np.int64), H.indices.astype(np.int64), H.data.astype(np.float64), norm2, yp.astype(np.int64),
            perms, tr, ev, S_h, loss, np.array([g[0] for g in gs]), np.array([g[1] for g in gs]), out)
    np.savez(fn, out=out, subs=np.array(subs), etas=np.array([g[0] for g in gs]), lams=np.array([g[1] for g in gs]),
             seconds=time.time() - t0, nnz_mean=H.nnz / H.shape[0])
    return str(fn), time.time() - t0


def run(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, names, up, mult = K.load_all()
    G.update(X=X, y=y, up=up, seeds=K.sketch_utils.sample_sketch_seeds(5))
    sizes = [int(((y == c1) | (y == c2)).sum()) for c1, c2 in up]
    pis = range(len(up)) if a.pairs is None else [int(v) for v in a.pairs.split(",")]
    shs = SH_GRID if a.sh is None else [int(v) for v in a.sh.split(",")]
    tasks = [(pi, S, si, l) for pi in pis for S in shs for si in range(a.seeds) for l in (0, 1)]
    tasks.sort(key=lambda t: -sizes[t[0]])     # biggest pairs first
    K.log(f"m1: {len(tasks)} tasks on {a.procs} procs")
    with mp.get_context("fork").Pool(a.procs) as pool:
        for i, (fn, s) in enumerate(pool.imap_unordered(task, tasks, chunksize=1)):
            if (i + 1) % 100 == 0 or i + 1 == len(tasks) or a.pairs:
                K.log(f"m1: {i + 1}/{len(tasks)} ({(time.time() - t0) / 60:.1f} min) last {s:.1f}s {fn.split('/')[-1]}")
    K.log(f"m1 done {(time.time() - t0) / 60:.1f} min, peak parent RSS {K.peak_ram_gb()} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--procs", type=int, default=112)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--sh", default=None)
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    run(ap.parse_args())
