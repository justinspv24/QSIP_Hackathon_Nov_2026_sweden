"""Tox21 sweep: machine size vs classification quality, imbalance-aware.

Extends the real-dataset sweeps of Zhao, Zlokapa, Preskill, Huang
(arXiv 2604.07639) to Tox21 toxicity screening with Morgan fingerprints.
Same pipeline as their dorothea_svm.py (ridge / LS-SVM, balanced class
weights, 5-fold CV), plus the metrics that matter for rare hits:
AUPRC, minority-class recall and balanced accuracy, not just accuracy.

For each fingerprint size D (feature hashing of a 2^20-bit Morgan FP):
    classical streaming memory = D
    classical sparse / QRAM    = nnz of the data matrix
    quantum oracle sketching   = 2 ceil(log2(N+2D)) + ceil(log2(s+1)) + 4

Experiment 2 (--imbalance): fix D, downsample the positives to lower hit
rates, and track how the quality metrics behave. Machine sizes barely move.

Usage (from this folder):
    pip install rdkit pandas scikit-learn matplotlib
    python tox21_svm.py                       # task SR-p53, D sweep
    python tox21_svm.py --task NR-AhR
    python tox21_svm.py --imbalance           # hit-rate sweep at D = 2048
Outputs: tox21_<task>_sweep.json/.png, tox21_<task>_imbalance.json/.png
"""
import argparse
import json
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, recall_score)
from sklearn.model_selection import StratifiedKFold

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
         "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
FULL_BITS = 2 ** 20
DIMS = [2 ** k for k in range(6, 21)]          # 64 ... 1,048,576
METRICS = ["accuracy", "balanced_accuracy", "auprc", "minority_recall"]


def qos_machine_size(n_samples, n_features, sparsity):
    """LS-SVM machine size of QOS (qubits + 1 float), as in sweep_utils.py."""
    return float(2 * np.ceil(np.log2(n_samples + 2 * n_features))
                 + np.ceil(np.log2(sparsity + 1)) + 4)


def load_tox21(task, cache="data_cache"):
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "tox21.csv.gz")
    if not os.path.exists(path):
        print(f"Downloading Tox21 from {URL} ...")
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path)
    df = df[df[task].notna()].reset_index(drop=True)   # drop missing labels
    return df["smiles"].tolist(), df[task].astype(int).to_numpy()


def morgan_onbits(smiles, radius=2):
    """On-bit indices of a 2^20-bit Morgan fingerprint per molecule."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=FULL_BITS)
    bits, keep = [], []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bits.append(np.fromiter(gen.GetFingerprint(mol).GetOnBits(), dtype=np.int64))
        keep.append(i)
    return bits, np.array(keep)


def fold_features(bits, D):
    """Hash the 2^20-bit fingerprint down to D bits (bit index mod D)."""
    rows, cols = [], []
    for r, b in enumerate(bits):
        c = np.unique(b % D)
        rows.append(np.full(c.size, r))
        cols.append(c)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(len(bits), D))


def evaluate(X, y, alpha, seed=42):
    """5-fold stratified CV; returns per-fold scores for each metric."""
    out = {m: [] for m in METRICS}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = RidgeClassifier(alpha=alpha, class_weight="balanced").fit(X[tr], y[tr])
        pred, score = clf.predict(X[te]), clf.decision_function(X[te])
        out["accuracy"].append(accuracy_score(y[te], pred))
        out["balanced_accuracy"].append(balanced_accuracy_score(y[te], pred))
        out["auprc"].append(average_precision_score(y[te], score))
        out["minority_recall"].append(recall_score(y[te], pred, pos_label=1))
    return {m: [float(v) for v in vals] for m, vals in out.items()}


def sizes(X):
    N, D = X.shape
    s = int(np.diff(X.indptr).max())
    return {"streaming": float(D), "sparse": float(X.nnz),
            "quantum": qos_machine_size(N, D, s), "max_sparsity": s}


def run_sweep(bits, y, task, alpha):
    res = {"task": task, "n_samples": len(y), "pos_rate": float(y.mean()),
           "alpha": alpha, "points": []}
    print(f"{task}: N = {len(y)}, positive rate = {y.mean():.2%}")
    for D in DIMS:
        X = fold_features(bits, D)
        pt = {"D": D, **sizes(X), **evaluate(X, y, alpha)}
        res["points"].append(pt)
        print(f"  D={D:>8d}  Q={pt['quantum']:.0f}  "
              f"AUPRC={np.mean(pt['auprc']):.3f}  "
              f"bal.acc={np.mean(pt['balanced_accuracy']):.3f}  "
              f"acc={np.mean(pt['accuracy']):.3f}")
    return res


def plot_sweep(res, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pts = res["points"]
    for ax, metric, label in [(axes[0], "accuracy", "Accuracy (paper's metric)"),
                              (axes[1], "auprc", "AUPRC (what matters for rare hits)")]:
        q = [np.mean(p[metric]) for p in pts]
        for key, color, name in [("sparse", "tab:blue", "Classical sparse / QRAM"),
                                 ("streaming", "tab:orange", "Classical streaming"),
                                 ("quantum", "tab:purple", "Quantum oracle sketching")]:
            ax.plot(q, [p[key] for p in pts], "o-", color=color, label=name, ms=4)
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Machine size (memory units)")
        ax.grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Tox21 {res['task']} (hit rate {res['pos_rate']:.1%}, N = {res['n_samples']})")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    print(f"Saved {fname}")


def run_imbalance(bits, y, task, alpha, D, rates, n_rep=5):
    """Downsample positives to target hit rates at fixed D."""
    X_full = fold_features(bits, D)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    res = {"task": task, "D": D, "alpha": alpha, "points": []}
    rng = np.random.default_rng(0)
    for rate in rates:
        n_pos = int(round(rate * len(neg) / (1 - rate)))
        if n_pos > len(pos) or n_pos < 10:
            print(f"  skip rate {rate:.2%} (needs {n_pos} positives)")
            continue
        reps = {m: [] for m in METRICS}
        for r in range(n_rep):
            idx = np.concatenate([neg, rng.choice(pos, n_pos, replace=False)])
            scores = evaluate(X_full[idx], y[idx], alpha, seed=r)
            for m in METRICS:
                reps[m].append(float(np.mean(scores[m])))
        pt = {"pos_rate": rate, "n_pos": n_pos, **sizes(X_full[idx]), **reps}
        res["points"].append(pt)
        print(f"  rate={rate:6.2%}  n_pos={n_pos:4d}  AUPRC={np.mean(reps['auprc']):.3f}  "
              f"recall={np.mean(reps['minority_recall']):.3f}  acc={np.mean(reps['accuracy']):.3f}")
    return res


def plot_imbalance(res, fname):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    pts = res["points"]
    r = [p["pos_rate"] for p in pts]
    for m, c in [("accuracy", "gray"), ("balanced_accuracy", "tab:green"),
                 ("auprc", "tab:red"), ("minority_recall", "tab:blue")]:
        mu = [np.mean(p[m]) for p in pts]
        sd = [np.std(p[m]) for p in pts]
        ax.errorbar(r, mu, yerr=sd, fmt="o-", color=c, label=m, ms=4, capsize=2)
    ax.plot(r, r, "k:", lw=1, label="AUPRC of random guessing")
    ax.set_xscale("log")
    ax.set_xlabel("Hit rate (fraction of positives)")
    ax.set_ylabel("Score")
    ax.set_title(f"Tox21 {res['task']}, D = {res['D']}: rarer hits")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    print(f"Saved {fname}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="SR-p53", choices=TASKS)
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge regularization")
    ap.add_argument("--imbalance", action="store_true", help="run the hit-rate sweep")
    ap.add_argument("--D", type=int, default=2048, help="fingerprint size for --imbalance")
    args = ap.parse_args()

    smiles, y_all = load_tox21(args.task)
    print("Computing Morgan fingerprints ...")
    bits, keep = morgan_onbits(smiles)
    y = y_all[keep]

    if args.imbalance:
        rates = [0.10, 0.05, 0.02, 0.01, 0.005, 0.002]
        res = run_imbalance(bits, y, args.task, args.alpha, args.D, rates)
        stem = f"tox21_{args.task}_imbalance"
        plot_imbalance(res, stem + ".png")
    else:
        res = run_sweep(bits, y, args.task, args.alpha)
        stem = f"tox21_{args.task}_sweep"
        plot_sweep(res, stem + ".png")
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"Saved {stem}.json")
