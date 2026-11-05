"""E1 — classical baselines: logistic regression (class-balanced) and the exact centroid (M -> infinity sketch),
for each rare class and each N. Also logistic regression on ALL genes (the honest 'classical wins' reference).
Writes results/e1_baselines.json."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.linear_model import LogisticRegression

from qos_cells import load_config
from qos_cells.load_data import load_pbmc3k, make_task, stratified_split
from qos_cells.sketch import exact_phases, scores_from_p
from qos_cells.readout import best_threshold, evaluate
from qos_cells.io import save_json


def logreg_eval(Xtr, ytr, Xte, yte, seed):
    clf = LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0, random_state=seed)
    clf.fit(Xtr, ytr)
    s_tr, s_te = clf.decision_function(Xtr), clf.decision_function(Xte)
    t = best_threshold(s_tr, ytr)
    return evaluate(s_te, yte, t)


def main():
    cfg = load_config()
    seed, tf, mc = cfg["seed"], cfg["data"]["test_frac"], cfg["data"]["min_cells_per_gene"]
    X, genes, labels, summary = load_pbmc3k(cfg["data"]["path"])
    print(f"[E1] {summary['n_cells']} cells x {summary['n_genes']} genes, {summary['fraction_zero']*100:.1f}% zeros")
    out = dict(dataset=summary, tasks=[])

    for positive in cfg["tasks"]["positives"]:
        y = (labels == positive).astype(int)
        tr, te = stratified_split(y, tf, seed)
        t0 = time.time()
        ref = logreg_eval(X[tr], y[tr], X[te], y[te], seed)       # all genes, log-expression features
        print(f"[E1] {positive:16s} all-genes logreg: bal.acc={ref['balanced_accuracy']:.3f} AUPRC={ref['auprc']:.3f} ({time.time()-t0:.1f}s)")
        rows = []
        for N in cfg["tasks"]["gene_counts"]:
            task = make_task(X, genes, labels, positive, N, seed, tf, mc)
            p_tr, p_te, y_tr, y_te = task["p_train"], task["p_test"], task["y_train"], task["y_test"]
            ab_tr, ab_te = task["abstain_train"], task["abstain_test"]
            # logistic regression on the same p features the quantum readout sees
            lr = logreg_eval(p_tr, y_tr, p_te, y_te, seed)
            # exact centroid = the M -> infinity sketch: phi ∝ mu1 - mu0, scores = p . sin(phi)
            phi, c = exact_phases(p_tr[~ab_tr], y_tr[~ab_tr], cfg["sketch"]["phase_limit"])
            s_tr = scores_from_p(p_tr, phi, ab_tr)
            s_te = scores_from_p(p_te, phi, ab_te)
            thr = best_threshold(s_tr, y_tr)
            cen = evaluate(s_te, y_te, thr)
            rows.append(dict(N=N, genes=task["genes"][:16], n_train_pos=task["n_train_pos"], n_test_pos=task["n_test_pos"],
                             pi_hat=task["pi_hat"], abstain_rate_test=task["abstain_rate_test"],
                             abstain_rate_test_pos=task["abstain_rate_test_pos"],
                             logreg_p=lr, exact_centroid=cen, threshold=thr, phase_scale_c=c,
                             stream_density=float((p_tr[~ab_tr] > 0).mean())))
            print(f"[E1]   N={N:5d}  centroid bal.acc={cen['balanced_accuracy']:.3f} AUPRC={cen['auprc']:.3f} | "
                  f"logreg(p) bal.acc={lr['balanced_accuracy']:.3f} AUPRC={lr['auprc']:.3f} | abstain {task['abstain_rate_test']*100:.1f}% "
                  f"(pos {task['abstain_rate_test_pos']*100:.1f}%) | genes {task['genes'][:4]}")
        out["tasks"].append(dict(positive=positive, base_rate=float(y.mean()), n_pos=int(y.sum()),
                                 logreg_all_genes=ref, by_N=rows))
    save_json("e1_baselines.json", out, cfg)


if __name__ == "__main__":
    main()
