"""Markdown tables for README / SLIDE_NOTES, from results/e13_stage*.json only.   python e13_tables.py 1|2|3|4"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
SH = {"CD4+ T Helper2": "Th2", "CD34+": "CD34+", "CD4+/CD45RA+/CD25- Naive T": "naive CD4 T", "CD19+ B": "B",
      "Megakaryocytes": "Mk", "Dendritic cells": "DC", "B cells": "B"}
M = ["count_sketch", "hashing", "jl_sparse", "heavy_hitters_mg", "heavy_hitters_top", "hashing_logreg"]


def f(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, float) and (abs(x) < 1e-2 and x != 0):
        return f"{x:.1e}"
    return f"{x:.{d}f}" if isinstance(x, float) else f"{x:,}"


def stage_tasks(st):
    d = json.loads((RES / f"e13_stage{st}.json").read_text())
    return d, sorted(d["tasks"], key=lambda t: (t["N"], t["positive"]))


def t1(st=1):
    d, ts = stage_tasks(st)
    print(f"partial={d['partial']}  jobs={d.get('n_jobs_done', len(ts))}")
    print("| features | N | qubits | task | π̂ | test pos | exact BA | logreg BA | AUPRC exact | d_eff | M_run/N | best one-pass | S* | S*/N | bytes | S*/N (M→∞) | S*/N tol 0.02 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in ts:
        b = t["best_one_pass"]
        tol2 = [m.get("S_star_tol002") for k, m in t["methods"].items() if k in M[:4] and m.get("S_star_tol002")]
        fs = f"{t['feature_set']} K={t['K']}"
        print(f"| {fs} | {t['N']:,} | {t['index_qubits']} | {SH.get(t['positive'], t['positive'])}{'' if t.get('signal', True) else ' (no signal)'} | "
              f"{t['pi_hat']:.4f} | {t['n_test_pos']} | {t['exact_centroid']['balanced_accuracy']:.3f} | {f(t.get('logreg_full', {}).get('balanced_accuracy'))} | "
              f"{t['exact_centroid']['auprc']:.3f} | {t['d_eff']:.0f} | {t['M_run_over_N']:,} | {b['method']} | {f(b['S_star'])} | {f(b['S_star_over_N'])} | "
              f"{f(b['bytes'])} | {f(b['Minf']['S_star_over_N'])} | {f(min(tol2) / t['N'] if tol2 else None)} |")
    print()
    print("S* per method (stream at M_run; stable rule; — = never inside the band):")
    print("| features | task | " + " | ".join(M) + " |")
    print("|---|---|" + "---|" * len(M))
    for t in ts:
        print(f"| {t['feature_set']} K={t['K']} | {SH.get(t['positive'])} | " + " | ".join(f(t["methods"].get(k, {}).get("S_star")) for k in M) + " |")


def t2():
    d = json.loads((RES / "e13_stage2.json").read_text())
    print(f"partial={d['partial']} jobs={d['n_jobs_done']}/{d['n_jobs_planned']}")
    print("| dataset | features | task | splits | exact BA (mean ± sd) | best one-pass S*/N mean ± sd | min–max | not reached | S*/N (M→∞) mean ± sd |")
    print("|---|---|---|---|---|---|---|---|---|")
    import numpy as np
    for g in d["robustness"]:
        b, bi = g["best_S_star_over_N"], g["best_S_star_over_N_Minf"]
        nr = sum(x is None for x in g["best_S_star_over_N_per_split"])
        print(f"| {g['dataset']} | {g['feature_set']} | {SH.get(g['positive'])}{'' if g['signal'] else ' (no signal)'} | {g['n_splits']} | "
              f"{np.mean(g['exact_BA']):.3f} ± {np.std(g['exact_BA']):.3f} | {f(b['mean'])} ± {f(b['std'])} | {f(b['min'])}–{f(b['max'])} | {nr} | "
              f"{f(bi['mean'])} ± {f(bi['std'])} |")
    print()
    print("Per method S*/N, mean ± sd over splits (stream, stable rule; never reached = 1):")
    print("| dataset | features | task | " + " | ".join(M) + " |")
    print("|---|---|---|" + "---|" * len(M))
    for g in d["robustness"]:
        print(f"| {g['dataset']} | {g['feature_set']} | {SH.get(g['positive'])} | " +
              " | ".join((f"{f(g['methods'][k]['mean'])} ± {f(g['methods'][k]['std'])}" if k in g["methods"] else "—") for k in M) + " |")


def t4():
    d = json.loads((RES / "e13_stage4.json").read_text())
    print(f"partial={d['partial']} D={d['D']} pairs={d['n_pairs']} full ridge acc {d['full_ridge']['mean']:.4f} ± {d['full_ridge']['sem']:.4f}; "
          f"exact centroid acc {d['exact_centroid']['accuracy']['mean']:.4f}; quantum machine size (paper formula) {d['quantum']['paper_machine_size_mean']:.1f}, "
          f"log2 D = {d['quantum']['index_qubits']}")
    for k, rows in d["curves"].items():
        print(k, " ".join(f"{r['S']:.0f}:{r['accuracy_mean']:.3f}" for r in rows))
    print("S* (accuracy within 0.01 of the reference, all larger S too):", d["S_star_tol001"])


if __name__ == "__main__":
    a = sys.argv[1]
    {"1": t1, "3": lambda: t1(3), "2": t2, "4": t4}[a]()
