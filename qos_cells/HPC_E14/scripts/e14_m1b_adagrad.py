"""E14 method 1b: signed hashing + AdaGrad SGD (Vowpal Wabbit's default adaptive per-coordinate step).
Setup check for M1: is the plain normalised step leaving accuracy on the table?
State: S weights + S squared-gradient accumulators + bias (+ its accumulator) + 2 scalars.
  (a) = 2S (accumulators are learner state, counted as registers)   (b) = 2S + 4
Update on sample i (balanced weight c = n/(2 n_class), known class counts as in M1):
  p = w.x + b; g = c (p - y) [squared] or -c y sigmoid(-y p) [logistic]
  for j in nz(x): G_j += (g x_j)^2 ; w_j -= eta0 (g x_j + lam w_j) / sqrt(G_j + 1e-8)   (L2 on touched coords, VW-style)
  bias likewise (no L2). Epochs 5, snapshot after 1 and 5. Same tasks / subsets / output layout as M1 (no averaged copy:
  the avg axis holds the same weights twice, so the merge treats it like M1 with avg = plain).

    python e14_m1b_adagrad.py run --procs 120
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(__file__))
import e14_core as K  # noqa: E402
import e14_m1_sgd as M1  # noqa: E402

ETAS = [0.01, 0.03, 0.1, 0.3]
LAMS = [0.0, 1e-4, 1e-3, 1e-2]


@njit(cache=True)
def ada_all(indptr, ind, val, norm2, y, perms, tr, ev, S, loss, etas, lams, out):
    U = tr.shape[0]
    for u in range(U):
        ntot = 0
        npos = 0
        for i in range(len(y)):
            if tr[u, i]:
                ntot += 1
                npos += y[i]
        cpos = ntot / (2.0 * npos)
        cneg = ntot / (2.0 * (ntot - npos))
        for g in range(len(etas)):
            eta0 = etas[g]
            lam = lams[g]
            w = np.zeros(S)
            G = np.zeros(S)
            b = 0.0
            Gb = 0.0
            si = 0
            E = perms.shape[0]
            for e in range(E):
                for ii in range(perms.shape[1]):
                    i = perms[e, ii]
                    if not tr[u, i]:
                        continue
                    if y[i] == 1:
                        c = cpos
                        ys = 1.0
                    else:
                        c = cneg
                        ys = -1.0
                    p = b
                    for q in range(indptr[i], indptr[i + 1]):
                        p += w[ind[q]] * val[q]
                    if loss == 0:
                        gr = c * (p - ys)
                    else:
                        m = ys * p
                        gr = 0.0 if m > 30 else -c * ys / (1.0 + np.exp(m))
                    for q in range(indptr[i], indptr[i + 1]):
                        j = ind[q]
                        gj = gr * val[q] + lam * w[j]
                        G[j] += gj * gj
                        w[j] -= eta0 * gj / np.sqrt(G[j] + 1e-8)
                    Gb += gr * gr
                    b -= eta0 * gr / np.sqrt(Gb + 1e-8)
                if si < 2 and e + 1 == (1 if si == 0 else E):
                    M1._eval(indptr, ind, val, y, ev[u], w, b, out[u, g, si, 0])
                    out[u, g, si, 1, :] = out[u, g, si, 0, :]
                    si += 1


def mem(S_h, avg):
    return 2 * S_h, 2 * S_h + 4


if __name__ == "__main__":
    # reuse M1's driver with the AdaGrad kernel and grid
    M1.sgd_all = ada_all
    M1.ETAS, M1.LAMS = ETAS, LAMS
    M1.OUT = K.E14 / "parts" / "m1b"
    M1.KNOWN_COUNTS = True
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--procs", type=int, default=120)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--sh", default=None)
    ap.add_argument("--seeds", type=int, default=M1.N_SEEDS)
    M1.run(ap.parse_args())
