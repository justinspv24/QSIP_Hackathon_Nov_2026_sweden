"""Gene-pair feature space without materialising per-cell pair vectors (GPU, CuPy).

Pair features (as in E7, scripts/09_route2_classical_memory.py):
    z_(g,h) = x_g x_h, g <= h over K genes;   p_(g,h) = z^2 / sum z^2 = x_g^2 x_h^2 / Z(x),
    Z(x) = sum_{g<=h} x_g^2 x_h^2 = ((sum x^2)^2 + sum x^4) / 2          (closed form; checked vs E7 code)
With u = x^2 / sqrt(Z):  p_(g,h) = u_g u_h  (g < h)  and  p_(g,g) = u_g^2.
Feature index j(g,h) = g K - g(g-1)/2 + (h-g)  = row-major upper triangle incl. diagonal (= T[triu_mask]).

Consequences used everywhere:
  * any linear score  sum_j p_ij D_j  =  u_i^T T u_i  with T = upper-triangular K x K matrix holding D;
  * class mean of p   =  triu( U^T U / n )   (a K x K Gram matrix);
  * logistic-regression gradient in pair space = triu( U^T diag(r) U ).
The record stream is the E7 Poissonised uniform stream: each (cell, pair) entry gets Poisson(lambda) records,
lambda = M / (N n_train). Equivalently (Poisson splitting) the total number of records on the nonzero entries is
Poisson(lambda * E) and each record hits a uniformly random nonzero entry: cell chosen with prob. m_i(m_i+1)/2 / E,
then a uniform pair (a <= b) of the cell's m_i nonzero genes. Same distribution, never materialised.
"""
from __future__ import annotations

import math

import cupy as cp
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------------------------------------------
# hashing (identical constants/semantics to E7's numpy _mix / hbucket / hsign)
# ---------------------------------------------------------------------------------------------------------------
_C1, _C2, _C3 = cp.uint64(0x9E3779B97F4A7C15), cp.uint64(0xBF58476D1CE4E5B9), cp.uint64(0x94D049BB133111EB)


def mix(j, seed):
    x = (j.astype(cp.uint64) ^ cp.uint64(int(seed))) * _C1
    x ^= x >> cp.uint64(30)
    x *= _C2
    x ^= x >> cp.uint64(27)
    x *= _C3
    x ^= x >> cp.uint64(31)
    return x


def hbucket(j, seed, B):
    return (mix(j, seed) % cp.uint64(B)).astype(cp.int64)


def hsign(j, seed):
    return 1.0 - 2.0 * (mix(j, seed) >> cp.uint64(63)).astype(cp.float64)


# ---------------------------------------------------------------------------------------------------------------
# data container
# ---------------------------------------------------------------------------------------------------------------
class PairData:
    """Xsel: scipy sparse (cells x K) expression on the K selected genes (rows = all cells of the task)."""

    def __init__(self, Xsel, dense_dtype=cp.float32):
        X = sp.csr_matrix(Xsel, dtype=np.float64)
        X.sort_indices()
        self.n, self.K = X.shape
        self.N = self.K * (self.K + 1) // 2
        sq = X.multiply(X).tocsr()
        sq.sort_indices()
        s2 = np.asarray(sq.sum(1)).ravel()
        s4 = np.asarray(sq.multiply(sq).sum(1)).ravel()
        Z = (s2**2 + s4) / 2.0
        self.Z = Z
        self.abstain = Z == 0
        inv = np.zeros_like(Z)
        inv[~self.abstain] = 1.0 / np.sqrt(Z[~self.abstain])
        Uc = sp.diags(inv) @ sq
        Uc = sp.csr_matrix(Uc)
        Uc.sort_indices()
        self.U_csr = Uc
        self.m = np.diff(Uc.indptr).astype(np.int64)                       # nonzero genes per cell
        self.npairs = self.m * (self.m + 1) // 2                            # nonzero pair entries per cell
        # q_i = sum_j p_ij^2 = ((sum u^2)^2 + sum u^4)/2  (for the closed-form stream variance)
        u2 = np.asarray(Uc.multiply(Uc).sum(1)).ravel()
        u4 = np.asarray(Uc.multiply(Uc).multiply(Uc).multiply(Uc).sum(1)).ravel()
        self.q = (u2**2 + u4) / 2.0
        self.Ud = cp.asarray(Uc.toarray(), dtype=dense_dtype)              # dense u on GPU (scoring, Gram)
        self.indptr = cp.asarray(Uc.indptr.astype(np.int64))
        self.indices = cp.asarray(Uc.indices.astype(np.int64))
        self.data = cp.asarray(Uc.data.astype(np.float64))
        self._mask = None

    # ---- triangular helpers ----------------------------------------------------------------------------------
    @property
    def mask(self):
        if self._mask is None:
            self._mask = cp.triu(cp.ones((self.K, self.K), dtype=cp.bool_))
        return self._mask

    def to_T(self, D, dtype=cp.float32):
        T = cp.zeros((self.K, self.K), dtype=dtype)
        T[self.mask] = cp.asarray(D, dtype=dtype)
        return T

    def from_T(self, T):
        return T[self.mask]

    def pair_index(self, g, h):
        g = g.astype(cp.int64)
        h = h.astype(cp.int64)
        return g * self.K - g * (g - 1) // 2 + (h - g)

    # ---- class statistics -------------------------------------------------------------------------------------
    def gram(self, rows, weights=None):
        """sum_i w_i u_i u_i^T over rows (float64 accumulate via float32 GEMM blocks)."""
        rows = np.asarray(rows)
        G = cp.zeros((self.K, self.K), dtype=cp.float64)
        wt = None if weights is None else np.asarray(weights, dtype=np.float32)
        for a in range(0, len(rows), 100_000):                      # row batches: no full-size copies
            U = self.Ud[cp.asarray(rows[a:a + 100_000])].astype(cp.float32)
            Uw = U if wt is None else U * cp.asarray(wt[a:a + 100_000])[:, None]
            G += (Uw.T @ U).astype(cp.float64)
            del U, Uw
        return G

    def centroid_delta(self, rows, y):
        """Exact centroid Delta = mean_pos p - mean_neg p over `rows` (pair space, length N, float64)."""
        rows = np.asarray(rows)
        y = np.asarray(y)
        r1, r0 = rows[y == 1], rows[y == 0]
        G = self.gram(r1) / len(r1) - self.gram(r0) / len(r0)
        return self.from_T(G)

    # ---- scoring ---------------------------------------------------------------------------------------------
    def scores(self, D, rows=None, batch=200_000):
        """s_i = sum_j p_ij D_j for every cell (or `rows`). D: (N,) array. Returns float64 cupy array."""
        D = cp.asarray(D)
        mx = float(cp.abs(D).max()) or 1.0
        T = self.to_T(D / mx, cp.float32)
        U = self.Ud if rows is None else self.Ud[cp.asarray(rows)]
        out = cp.empty(U.shape[0], dtype=cp.float64)
        for a in range(0, U.shape[0], batch):
            Ub = U[a:a + batch].astype(cp.float32)
            out[a:a + batch] = cp.sum((Ub @ T) * Ub, axis=1, dtype=cp.float64)
        del T
        return out * mx

    # ---- explicit pair enumeration (gate checks, hashed features) --------------------------------------------
    def cell_pairs(self, cells):
        """All nonzero pair entries of the given cells: (local cell id, j, p). Cupy arrays."""
        cells = np.asarray(cells)
        m = self.m[cells]
        npairs = m * (m + 1) // 2
        tot = int(npairs.sum())
        off = np.zeros(len(cells) + 1, dtype=np.int64)
        off[1:] = np.cumsum(npairs)
        cid = cp.repeat(cp.arange(len(cells), dtype=cp.int64), cp.asarray(npairs))
        k = cp.arange(tot, dtype=cp.int64) - cp.asarray(off[:-1])[cid]
        mm = cp.asarray(m)[cid].astype(cp.float64)
        # invert k -> (a, b), a <= b, row-major upper triangle of an m x m matrix
        a = cp.floor(((2 * mm + 1) - cp.sqrt((2 * mm + 1) ** 2 - 8 * k)) / 2).astype(cp.int64)
        mi = cp.asarray(m)[cid]
        start = a * mi - a * (a - 1) // 2
        a = cp.where(start > k, a - 1, a)
        start = a * mi - a * (a - 1) // 2
        nxt = (a + 1) * mi - (a + 1) * a // 2
        a = cp.where(k >= nxt, a + 1, a)
        start = a * mi - a * (a - 1) // 2
        b = k - start + a
        base = self.indptr[cp.asarray(cells)][cid]
        ga, gb = self.indices[base + a], self.indices[base + b]
        p = self.data[base + a] * self.data[base + b]
        return cid, self.pair_index(ga, gb), p

    # ---- the record stream -----------------------------------------------------------------------------------
    def stream(self, rows, w_cell, M, seed, batch=50_000_000, n_records_override=None):
        """Poissonised uniform stream over the entries of the cells `rows` with per-cell weights w_cell (aligned with
        rows). Returns (phi, phi_pos, n_records): phi = (N/M) sum c_ij w_i p_ij;  phi_pos = rare-class part.
        If n_records_override is given, exactly that many records are drawn (reservoir) and the raw sum is returned."""
        rows = np.asarray(rows)
        npairs = self.npairs[rows].astype(np.float64)
        E = float(npairs.sum())
        cum = cp.asarray(np.concatenate([[0.0], np.cumsum(npairs)]))
        rng_host = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
        if n_records_override is None:
            lam = M / (self.N * len(rows))
            R = int(rng_host.poisson(lam * E))
        else:
            R = int(n_records_override)
        rs = cp.random.RandomState(int(rng_host.integers(0, 2**31 - 1)))
        rows_g = cp.asarray(rows)
        w_g = cp.asarray(w_cell, dtype=cp.float64)
        mrow = cp.asarray(self.m[rows])
        pos = cp.zeros(self.N, dtype=cp.float64)
        neg = cp.zeros(self.N, dtype=cp.float64)
        done = 0
        while done < R:
            B = min(batch, R - done)
            k = cp.floor(rs.random_sample(B) * E)
            ci = cp.searchsorted(cum, k, side="right") - 1                      # local cell index in rows
            ci = cp.clip(ci, 0, len(rows) - 1)
            mi = mrow[ci]
            r = cp.floor(rs.random_sample(B) * mi).astype(cp.int64)
            c = cp.floor(rs.random_sample(B) * (mi + 1)).astype(cp.int64)
            low = c <= r
            a = cp.where(low, c, mi - c)
            b = cp.where(low, r, mi - 1 - r)
            base = self.indptr[rows_g[ci]]
            ga, gb = self.indices[base + a], self.indices[base + b]
            v = w_g[ci] * self.data[base + a] * self.data[base + b]
            j = self.pair_index(ga, gb)
            ispos = v > 0
            pos += cp.bincount(j[ispos], weights=v[ispos], minlength=self.N)
            neg += cp.bincount(j[~ispos], weights=-v[~ispos], minlength=self.N)
            done += B
            del k, ci, mi, r, c, low, a, b, base, ga, gb, v, j, ispos
        if n_records_override is not None:
            return pos - neg, pos, R
        scale = self.N / M
        return scale * (pos - neg), scale * pos, R


# ---------------------------------------------------------------------------------------------------------------
# sketches (GPU) : phi (N,) -> decoded D (N,) + memory ledger (E7 semantics)
# ---------------------------------------------------------------------------------------------------------------
F64, I32, SEED_BYTES = 8, 4, 8


def ledger(floats=0, ints=0, seeds=0):
    return dict(floats=int(floats), ints=int(ints), seeds=int(seeds), bytes=int(F64 * floats + I32 * ints + SEED_BYTES * seeds))


def sk_count_sketch(phi, S, seeds, jidx, rows=3):
    B = max(1, S // rows)
    est = []
    for r in range(rows):
        h, s = hbucket(jidx, seeds[2 * r], B), hsign(jidx, seeds[2 * r + 1])
        C = cp.bincount(h, weights=s * phi, minlength=B)
        est.append(s * C[h])
        del h, s, C
    a, b, c = est
    med = cp.maximum(cp.minimum(a, b), cp.minimum(cp.maximum(a, b), c))
    return med, ledger(floats=B * rows, seeds=2 * rows)


def sk_hashing(phi, S, seeds, jidx):
    h, s = hbucket(jidx, seeds[0], S), hsign(jidx, seeds[1])
    C = cp.bincount(h, weights=s * phi, minlength=S)
    return s * C[h], ledger(floats=S, seeds=2)


def sk_jl_sparse(phi, S, seeds, jidx, s_rows=4):
    C = cp.zeros(S)
    for k in range(s_rows):
        h, s = hbucket(jidx, seeds[2 * k], S), hsign(jidx, seeds[2 * k + 1])
        C += cp.bincount(h, weights=s * phi, minlength=S)
    dec = cp.zeros_like(phi)
    for k in range(s_rows):
        h, s = hbucket(jidx, seeds[2 * k], S), hsign(jidx, seeds[2 * k + 1])
        dec += s * C[h]
    return dec / s_rows, ledger(floats=S, seeds=2 * s_rows)


class MG:
    """Misra-Gries via the mergeable-summary rule; sorted copies precomputed once per stream."""

    def __init__(self, ppos, pneg):
        self.ppos, self.pneg = ppos, pneg
        self.sp = cp.sort(ppos)[::-1]
        self.sn = cp.sort(pneg)[::-1]

    def __call__(self, S):
        k = max(1, S // 2)
        N = len(self.ppos)
        if k >= N:
            return self.ppos - self.pneg, ledger(floats=2 * k, ints=2 * k)
        tp, tn = self.sp[k], self.sn[k]                                    # (k+1)-th largest
        return cp.maximum(self.ppos - tp, 0.0) - cp.maximum(self.pneg - tn, 0.0), ledger(floats=2 * k, ints=2 * k)


class TopS:
    def __init__(self, phi):
        self.phi = phi
        self.order = cp.argsort(-cp.abs(phi))

    def __call__(self, S):
        N = len(self.phi)
        if S >= N:
            return self.phi.copy(), ledger(floats=N, ints=N)
        out = cp.zeros_like(self.phi)
        keep = self.order[:S]
        out[keep] = self.phi[keep]
        return out, ledger(floats=S, ints=S)


# ---------------------------------------------------------------------------------------------------------------
# class-balanced L2 logistic regression (sklearn objective, lbfgs, gtol = tol), GPU forward/backward
# ---------------------------------------------------------------------------------------------------------------
def fit_logreg(forward, backward, n_w, y, C=1.0, max_iter=300, tol=1e-4):
    """Minimise (1/n) sum_i sw_i log(1+exp(-y_i z_i)) + ||w||^2 / (2 C n), z = forward(w) + b, sw balanced.
    This is sklearn's LogisticRegression(class_weight='balanced', C) objective up to a positive constant
    (sklearn >= 1.1 scales by 1/n the same way), so gtol has the same meaning."""
    from scipy.optimize import minimize

    y = np.asarray(y).astype(int)
    n = len(y)
    n1 = int(y.sum())
    sw = np.where(y == 1, n / (2.0 * n1), n / (2.0 * (n - n1)))
    sw_g = cp.asarray(sw)
    ys = cp.asarray(2.0 * y - 1.0)
    lam = 1.0 / (C * n)

    def f(x):
        w = cp.asarray(x[:-1])
        b = float(x[-1])
        z = forward(w) + b
        m = -ys * z
        loss = float(cp.sum(sw_g * cp.logaddexp(0.0, m))) / n + 0.5 * lam * float(w @ w)
        r = -(sw_g * ys / (1.0 + cp.exp(-m))) / n                         # dloss/dz
        gw = backward(r) + lam * w
        g = np.empty(n_w + 1)
        g[:-1] = cp.asnumpy(gw)
        g[-1] = float(r.sum())
        return loss, g

    # scipy's L-BFGS-B workspace is ~(2m+5) n doubles indexed with int32: m = 10 overflows (segfault) at n > ~8e7.
    maxcor = 10 if n_w < 5e7 else 5
    res = minimize(f, np.zeros(n_w + 1), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=max_iter, maxls=50, gtol=tol, ftol=64 * np.finfo(float).eps, maxcor=maxcor))
    return res.x[:-1], float(res.x[-1]), int(res.nit), bool(res.nit < max_iter)


def logreg_pairs(pd: PairData, tr_rows, y_tr, scale, C=1.0, max_iter=300, tol=1e-4):
    """Logistic regression on ALL N pair features, done in K x K form. Returns (w (N,), b, n_iter, converged)."""
    tr_rows = np.asarray(tr_rows)
    contiguous = bool(len(tr_rows) and tr_rows[0] == 0 and np.all(np.diff(tr_rows) == 1))
    U = pd.Ud[: len(tr_rows)] if contiguous else pd.Ud[cp.asarray(tr_rows)]   # a view when rows are 0..n-1
    B = 100_000

    def forward(w):
        T = pd.to_T(w, cp.float32)
        out = cp.empty(U.shape[0], dtype=cp.float64)
        for a in range(0, U.shape[0], B):
            Ub = U[a:a + B]
            out[a:a + B] = cp.sum((Ub @ T) * Ub, axis=1, dtype=cp.float64)
        return scale * out

    def backward(r):
        G = cp.zeros((pd.K, pd.K), dtype=cp.float32)
        r32 = r.astype(cp.float32)
        for a in range(0, U.shape[0], B):
            Ub = U[a:a + B]
            G += (Ub * r32[a:a + B, None]).T @ Ub
        return scale * pd.from_T(G).astype(cp.float64)

    return fit_logreg(forward, backward, pd.N, y_tr, C=C, max_iter=max_iter, tol=tol)


def logreg_dense(Xtr, y_tr, scale, C=1.0, max_iter=300, tol=1e-4):
    X = cp.asarray(Xtr, dtype=cp.float32)

    def forward(w):
        return scale * (X @ w.astype(cp.float32)).astype(cp.float64)

    def backward(r):
        return scale * (X.T @ r.astype(cp.float32)).astype(cp.float64)

    return fit_logreg(forward, backward, X.shape[1], y_tr, C=C, max_iter=max_iter, tol=tol)


def hashed_features(pd: PairData, cells, S_list, seeds, batch_pairs=40_000_000):
    """Signed feature hashing of every cell's pair vector into S buckets, for every S in S_list (same 2 seeds).
    Returns {S: cupy (len(cells), S) float32}."""
    cells = np.asarray(cells)
    out = {S: cp.zeros((len(cells), S), dtype=cp.float32) for S in S_list}
    npairs = pd.npairs[cells]
    a = 0
    while a < len(cells):
        b = a
        tot = 0
        while b < len(cells) and (tot + npairs[b] <= batch_pairs or b == a):
            tot += npairs[b]
            b += 1
        cid, j, p = pd.cell_pairs(cells[a:b])
        mx = mix(j, seeds[0])
        sg = (1.0 - 2.0 * (mix(j, seeds[1]) >> cp.uint64(63)).astype(cp.float64)) * p
        for S in S_list:
            h = (mx % cp.uint64(S)).astype(cp.int64)
            acc = cp.bincount(cid * S + h, weights=sg, minlength=(b - a) * S)
            out[S][a:b] = acc.reshape(b - a, S).astype(cp.float32)
        del cid, j, p, mx, sg
        a = b
    return out


def d_eff(D):
    D = cp.asarray(D, dtype=cp.float64)
    s2 = float(cp.sum(D**2))
    s4 = float(cp.sum(D**4))
    return s2 * s2 / s4 if s4 > 0 else float("nan")
