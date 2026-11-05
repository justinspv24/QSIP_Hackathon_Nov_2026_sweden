"""PBMC3k loader, stratified split, train-only gene selection and amplitude features.

Data flow
---------
X (cells x genes, log1p-normalised, sparse)  --select N genes on TRAIN only-->  x (cells x N, dense)
x  --amplitude encoding-->  a_g = x_g / ||x||,   p_g = a_g^2 = x_g^2 / ||x||^2   (sums to 1 per cell)

The quantum readout sees p (the squared amplitudes), so p is the feature vector of every classifier here.
Cells whose N selected genes are all zero have no quantum state (||x|| = 0): they are counted and sent to an
"abstain" bucket, never silently dropped.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import ROOT

PBMC3K_URL = "https://raw.githubusercontent.com/chanzuckerberg/cellxgene/main/example-dataset/pbmc3k.h5ad"


def fetch_pbmc3k(path: str | Path | None = None) -> Path:
    path = Path(path) if path else ROOT / "data" / "pbmc3k_processed.h5ad"
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[load_data] downloading PBMC3k (~25 MB) -> {path}")
        urllib.request.urlretrieve(PBMC3K_URL, path)
    return path


def load_pbmc3k(path=None):
    """Returns X (csr float32, cells x genes, log1p-normalised), gene names, cell-type labels, summary dict."""
    import anndata as ad
    import warnings

    p = fetch_pbmc3k(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = ad.read_h5ad(p)
    raw = adata.raw.to_adata() if adata.raw is not None else adata
    X = sp.csr_matrix(raw.X, dtype=np.float32)
    genes = np.asarray(raw.var_names, dtype=str)
    labels = np.asarray(adata.obs["louvain"].astype(str))
    counts = {k: int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    summary = dict(
        file=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()[:16],
        n_cells=int(X.shape[0]), n_genes=int(X.shape[1]), nnz=int(X.nnz),
        fraction_zero=float(1 - X.nnz / (X.shape[0] * X.shape[1])),
        genes_measured_in_10x_run=32738,
        class_counts=dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    )
    return X, genes, labels, summary


def stratified_split(y: np.ndarray, test_frac: float, seed: int):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(te)


def select_genes(Xtr, ytr, N: int, min_cells: int = 3) -> np.ndarray:
    """Top-N genes by |Delta_g| = |mu1_g - mu0_g| of the all-gene squared-amplitude features p = x^2/||x||^2,
    computed on the TRAIN split only.

    This is the quantity the sketch classifier's weights are proportional to, so the selection is aligned with the
    model: it picks the genes that carry the most class signal in the amplitude encoding (strong positive markers
    such as PPBP/PF4 for megakaryocytes, and broadly expressed genes the rare class lacks). It also keeps the
    abstain rate (all-zero cells) near zero, unlike a t-statistic on log-expression, which with 10 training
    positives can pick genes the rare class never expresses.
    """
    Xtr = sp.csr_matrix(Xtr)
    sq = Xtr.multiply(Xtr)
    norm2 = np.asarray(sq.sum(1)).ravel()
    norm2[norm2 == 0] = 1.0
    P = sp.diags(1.0 / norm2) @ sq
    delta = np.asarray(P[ytr == 1].mean(0)).ravel() - np.asarray(P[ytr == 0].mean(0)).ravel()
    n_cells = np.asarray((Xtr > 0).sum(0)).ravel()
    delta[n_cells < min_cells] = 0.0
    return np.argsort(-np.abs(delta), kind="stable")[:N]


def select_genes_welch(Xtr, ytr, N: int, min_cells: int = 3) -> np.ndarray:
    """Alternative (not used): top-N genes by |Welch t| on log-expression. Kept for the backup slide."""
    Xtr = sp.csr_matrix(Xtr)
    pos, neg = Xtr[ytr == 1], Xtr[ytr == 0]
    mp = np.asarray(pos.mean(0)).ravel()
    mn = np.asarray(neg.mean(0)).ravel()
    vp = np.asarray(pos.multiply(pos).mean(0)).ravel() - mp**2
    vn = np.asarray(neg.multiply(neg).mean(0)).ravel() - mn**2
    t = (mp - mn) / np.sqrt(np.maximum(vp, 0) / pos.shape[0] + np.maximum(vn, 0) / neg.shape[0] + 1e-9)
    n_cells = np.asarray((Xtr > 0).sum(0)).ravel()
    t[n_cells < min_cells] = 0.0
    order = np.argsort(-np.abs(t), kind="stable")
    return order[:N]


def amplitude_features(Xsel):
    """p = x^2 / ||x||^2 per cell (dense), plus the abstain mask (all-zero cells) and the dense x."""
    x = Xsel.toarray() if sp.issparse(Xsel) else np.asarray(Xsel, dtype=np.float64)
    x = x.astype(np.float64)
    sq = x**2
    norm2 = sq.sum(1)
    abstain = norm2 == 0
    p = np.zeros_like(sq)
    p[~abstain] = sq[~abstain] / norm2[~abstain, None]
    return p, abstain, x


def make_task(X, genes, labels, positive: str, N: int, seed: int, test_frac: float = 0.3, min_cells: int = 3):
    """Binary task: positive cell type vs everything else, N genes, stratified split. All choices use TRAIN only."""
    assert N & (N - 1) == 0, "N must be a power of 2 (it is a qubit register)"
    y = (labels == positive).astype(int)
    tr, te = stratified_split(y, test_frac, seed)
    g = select_genes(X[tr], y[tr], N, min_cells)
    p_tr, ab_tr, x_tr = amplitude_features(X[tr][:, g])
    p_te, ab_te, x_te = amplitude_features(X[te][:, g])
    pi_hat = float(y[tr].mean())
    return dict(
        positive=positive, N=int(N), seed=int(seed), genes=[str(s) for s in genes[g]], gene_idx=g,
        train_idx=tr, test_idx=te, y_train=y[tr], y_test=y[te],
        p_train=p_tr, p_test=p_te, x_train=x_tr, x_test=x_te,
        abstain_train=ab_tr, abstain_test=ab_te, pi_hat=pi_hat,
        n_train_pos=int(y[tr].sum()), n_test_pos=int(y[te].sum()),
        abstain_rate_test=float(ab_te.mean()), abstain_rate_test_pos=float(ab_te[y[te] == 1].mean()),
    )
