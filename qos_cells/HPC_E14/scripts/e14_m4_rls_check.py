"""E14 method 4 check: recursive least squares (weighted, one pass over the rows, on k genes chosen in a first pass)
gives the same classifier as exact balanced ridge on those genes. RLS on [x, 1] with P0 = diag(1/alpha I_k, 1/eps)
(eps = 1e-9: intercept effectively unregularised, as in RidgeClassifier) is the exact minimiser of
sum c_i (w.x_i + b - y_i)^2 + alpha |w|^2 + eps b^2.  State: P ((k+1)(k+2)/2 symmetric) + weights (k+1) + ids k.
Checked on several pairs, k and alpha: test-prediction agreement and accuracy difference vs e14_core.ridge_prefix.
Writes results/e14_m4_rls_check.json.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import e14_core as K  # noqa: E402


def rls(F, y, c, alpha, eps=1e-9):
    n, k = F.shape
    P = np.diag(np.r_[np.full(k, 1.0 / alpha), 1.0 / eps])
    w = np.zeros(k + 1)
    for i in range(n):
        x = np.r_[F[i], 1.0]
        Px = P @ x
        g = c[i] * Px / (1.0 + c[i] * x @ Px)
        w += g * ((2 * y[i] - 1) - x @ w)
        P -= np.outer(g, Px)
    return w


if __name__ == "__main__":
    t0 = time.time()
    X, y, names, up, mult = K.load_all()
    rows = []
    for pi in (13, 23, 0):
        Xp, yp = K.pair_data(X, y, up, pi)
        b = K.blocks(yp)
        tr, ev = b != 0, b == 0
        n1 = yp[tr].mean()
        delta = np.asarray(Xp[tr][yp[tr] == 1].mean(0)).ravel() - np.asarray(Xp[tr][yp[tr] == 0].mean(0)).ravel()
        for k in (8, 32):
            idx = np.argsort(-np.abs(delta), kind="stable")[:k]
            F = Xp[:, idx].toarray().astype(np.float64)
            ytr = yp[tr]
            c = np.where(ytr == 1, 1 / (2 * n1), 1 / (2 * (1 - n1)))
            perm = np.random.default_rng(0).permutation(tr.sum())
            for al in (10.0, 1000.0):
                w = rls(F[tr][perm], ytr[perm], c[perm], al)
                p_rls = ((F[ev] @ w[:-1] + w[-1]) > 0).astype(int)
                p_ex = K.ridge_prefix(np, F.astype(np.float32), yp, tr, ev, [k], alphas=[al])[(k, al)]
                rows.append(dict(pair=pi, k=k, alpha=al, agree=float((p_rls == p_ex).mean()),
                                 acc_rls=K.acc_bacc(yp[ev], p_rls)[0], acc_exact=K.acc_bacc(yp[ev], p_ex)[0]))
                print(rows[-1], flush=True)
    body = dict(schema_version=1, experiment="e14_m4_rls_check", git_commit=K.git_commit(), hardware=K.hardware(),
                note="selection by exact |Delta| on the training fold (fold 0); RLS = exact ridge up to float error / eps",
                rows=rows, max_abs_acc_diff=max(abs(r["acc_rls"] - r["acc_exact"]) for r in rows),
                min_agreement=min(r["agree"] for r in rows), wall_seconds=round(time.time() - t0, 1), peak_ram_gb=K.peak_ram_gb())
    K.write_json(K.E14 / "results" / "e14_m4_rls_check.json", body)
    print("max |acc diff|", body["max_abs_acc_diff"], "min agreement", body["min_agreement"])
