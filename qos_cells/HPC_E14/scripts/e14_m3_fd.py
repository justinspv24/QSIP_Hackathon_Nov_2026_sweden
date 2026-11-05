"""E14 method 3: signed hashing to d dims (paper 'jl' sketch, paper seeds 0-2) + Frequent Directions (Liberty 2013,
Ghashami et al. 2016) sketch of rank ell of the weighted row stream, + weighted sums; ridge with the low-rank + alpha I
covariance. One pass. Rows are centred in-stream by Welford's update (exact centred weighted scatter, no extra memory
beyond the running sum of c x that is kept anyway); robust-FD correction adds half the total shrinkage as delta*I.

Stream (training rows of a subset, seeded random order): row weight c = n / (2 n_class), the paper's balanced weight
(class counts known up front: label-only pre-count, 2 integers). FD state B (ell x d): insert ell/2 rows sqrt(c_t) x_t into the empty half, then SVD and shrink all
squared singular values by the (ell/2+1)-th largest (top ell/2 rows survive) -- the ell-row FD variant (memory ell*d).
Also kept: sum c x (d), sum c y x (d), sum c, sum c y, counters.
Solve: Cov = B^T B - (sum c) xbar xbar^T ; w = (Cov + alpha I)^-1 (sum c y x - sum c * ybar * xbar); b = ybar - xbar.w.
(The d x d solve is a simulation convenience; a learner would use the SVD of B (= its own ell x d memory) and Woodbury.)
Memory: (a) paper accounting = d (model coefficients; covariance / its sketch not counted, as for exact ridge on d
hashed features);  (b) true = ell*d + 2d + 3 (+ d weights after the solve overwrite one d-vector -> same).
Grid: for every budget S and ell in L_GRID, d = floor((S-3)/(ell+2)) (d >= ell). alpha by nested CV.

    python e14_m3_fd.py --gpus 8
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e14_core as K  # noqa: E402

OUT = K.E14 / "parts" / "m3"
N_SEEDS = 3
L_GRID = [2, 4, 8, 16, 32, 64]
# robust FD (Luo et al. 2019): optionally add half the accumulated shrinkage back as delta*I (1 extra float);
# on/off is tuned with alpha by nested CV (last axis: 5 alphas plain, then 5 alphas with RFD)


def configs():
    cs = []
    for S in K.S_GRID:
        for ell in L_GRID:
            d = (S - 3) // (ell + 2)
            if d >= ell and d >= 2:
                cs.append((d, ell))
    return sorted(set(cs))


def mem(d, ell):
    return d, ell * d + 2 * d + 4


def fd_ridge_batched(cp, Fcat, blocks_, y, perm, trm, evm, ell, alphas):
    """All (seed, d) blocks with the same ell share one stream: Fcat (n, W) = concatenated hashed blocks; FD runs
    independently per block (per-block Grams via a block indicator). Returns acc, bacc (U, Q, n_alpha)."""
    U = trm.shape[0]
    n, Wd = Fcat.shape
    Q = len(blocks_)
    qid = np.concatenate([np.full(e - s0, q) for q, (s0, e) in enumerate(blocks_)])
    qg = cp.asarray(qid)
    Ind = cp.zeros((Wd, Q), cp.float32)
    Ind[cp.arange(Wd), qg] = 1
    Fp = cp.concatenate([Fcat, cp.zeros((1, Wd), Fcat.dtype)])
    streams = [perm[trm[u][perm]] for u in range(U)]
    h = ell // 2
    Lmax = max(len(s_) for s_ in streams)
    Lmax = ((Lmax + h - 1) // h) * h
    IDX = np.full((U, Lmax), n, np.int64)
    W = np.zeros((U, Lmax), np.float32)
    for u, s_ in enumerate(streams):
        IDX[u, :len(s_)] = s_
        ys = y[s_]
        n1 = ys.sum()
        W[u, :len(s_)] = len(s_) / (2.0 * np.where(ys == 1, n1, len(s_) - n1))   # paper's balanced weights (counts known)
    IDXg, Wg = cp.asarray(IDX), cp.asarray(W)
    # Welford: row sqrt(c_t W_{t-1}/W_t) (x_t - m_{t-1}) => sum of row row^T = exact centred weighted scatter
    Wcum = np.cumsum(W, 1)
    Wprev = np.concatenate([np.zeros((U, 1), np.float32), Wcum[:, :-1]], 1)
    coef = cp.asarray(np.sqrt(np.where(Wcum > 0, W * Wprev / np.maximum(Wcum, 1e-30), 0)).astype(np.float32))
    run = cp.zeros((U, Wd), cp.float32)          # running sum c x  (= the counted sum-of-x vector)
    Wpg = cp.asarray(Wprev)
    shrink = cp.zeros((U, Q), cp.float64)        # total FD shrinkage per block (robust FD correction)
    B = cp.zeros((U, ell, Wd), cp.float32)
    for st in range(0, Lmax, h):
        Xc = Fp[IDXg[:, st:st + h]]                                       # (U, h, W)
        cx = Xc * Wg[:, st:st + h, None]
        before = run[:, None, :] + cp.cumsum(cx, 1) - cx                  # sum c x over rows before each row
        mprev = before / cp.maximum(Wpg[:, st:st + h, None], 1e-30)
        B[:, h:] = (Xc - mprev) * coef[:, st:st + h, None]
        run += cx.sum(1)
        Kg = ((B[:, :, None, :] * B[:, None, :, :]) @ Ind).transpose(0, 3, 1, 2)    # (U, Q, ell, ell)
        lam, V = cp.linalg.eigh(Kg.astype(cp.float64))
        delta = lam[:, :, ell - 1 - h][:, :, None]
        shrink += cp.maximum(delta[:, :, 0], 0)
        top = lam[:, :, ell - h:]
        fac = cp.sqrt(cp.maximum(top - delta, 0) / cp.maximum(top, 1e-30)).astype(cp.float32)   # (U, Q, h)
        Vt = V[:, :, :, ell - h:].astype(cp.float32)                                           # (U, Q, ell, h)
        Vc = Vt[:, qg]                                                                          # (U, W, ell, h)
        newB = cp.einsum("uwik,uiw->ukw", Vc, B) * fac[:, qg].transpose(0, 2, 1)
        B[:, :h] = newB
        B[:, h:] = 0
    Ys = cp.concatenate([(2 * cp.asarray(y) - 1).astype(cp.float32), cp.zeros(1, cp.float32)])
    yg = cp.asarray(y)
    acc = np.zeros((U, Q, 2 * len(alphas)), np.float32)     # [alphas without RFD, alphas with RFD]
    bacc = np.zeros_like(acc)
    for u in range(U):
        idx, w = IDXg[u], Wg[u]
        sw = float(w.sum())
        ys_s = Ys[idx]
        ybar = float((w * ys_s).sum() / sw)
        X = Fp[idx]
        xbar_all = (w[:, None] * X).sum(0) / sw
        r_all = ((w * ys_s)[:, None] * X).sum(0) - sw * ybar * xbar_all
        ev = cp.asarray(evm[u])
        ye = cp.asnumpy(yg[ev])
        Fe_all = Fcat[ev]
        for q, (s0, e) in enumerate(blocks_):
            d = e - s0
            Bq = B[u, :, s0:e].astype(cp.float64)
            xb = xbar_all[s0:e].astype(cp.float64)
            Cov0 = Bq.T @ Bq
            rv = r_all[s0:e].astype(cp.float64)
            Fe = Fe_all[:, s0:e].astype(cp.float64)
            for ai, (al, rfd) in enumerate([(a_, 0) for a_ in alphas] + [(a_, 1) for a_ in alphas]):
                wv = cp.linalg.solve(Cov0 + (al + rfd * float(shrink[u, q]) / 2) * cp.eye(d), rv)
                pred = cp.asnumpy(((Fe @ wv + (ybar - float(xb @ wv))) > 0).astype(cp.int64))
                acc[u, q, ai], bacc[u, q, ai] = K.acc_bacc(ye, pred)
    return acc, bacc


def worker(gpu, q, X, y, up):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import cupy as cp
    seeds = K.sketch_utils.sample_sketch_seeds(5)
    cfg = configs()
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
        acc = np.full((N_SEEDS, len(cfg), len(subs), 2 * len(K.ALPHAS)), np.nan, np.float32)
        bacc = np.full_like(acc, np.nan)
        perm = np.random.default_rng([31, pi]).permutation(len(yp))
        for ell in L_GRID:
            items = [(si, ci, d) for si in range(N_SEEDS) for ci, (d, l) in enumerate(cfg) if l == ell]
            if not items:
                continue
            mats, blocks_, s0 = [], [], 0
            for si, ci, d in items:
                mats.append(K.hashed(Xp, d, seeds[si]).toarray())
                blocks_.append((s0, s0 + d))
                s0 += d
            Fcat = cp.asarray(np.concatenate(mats, 1))
            a_, b_ = fd_ridge_batched(cp, Fcat, blocks_, yp, perm, trm, evm, ell, K.ALPHAS)
            for qi, (si, ci, d) in enumerate(items):
                acc[si, ci], bacc[si, ci] = a_[:, qi], b_[:, qi]
            del Fcat
        np.savez(fn, acc=acc, bacc=bacc, subs=np.array(subs), cfg=np.array(cfg), seconds=time.time() - t0)
        K.log(f"m3 gpu{gpu}: pair {pi} n={len(yp)} {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--pairs", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, names, up, mult = K.load_all()
    sizes = [int(((y == c1) | (y == c2)).sum()) for c1, c2 in up]
    pis = list(range(len(up))) if a.pairs is None else [int(v) for v in a.pairs.split(",")]
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    for p in sorted(pis, key=lambda p: -sizes[p]):
        q.put(p)
    for _ in range(a.gpus):
        q.put(None)
    K.log(f"m3: {len(pis)} pairs, {len(configs())} (d, ell) configs x {N_SEEDS} seeds x 25 subsets")
    ps = [ctx.Process(target=worker, args=(g, q, X, y, up)) for g in range(a.gpus)]
    [p.start() for p in ps]
    [p.join() for p in ps]
    K.log("m3 done")
