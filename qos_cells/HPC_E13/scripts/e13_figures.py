"""E13 figures, built from JSON only.

    python e13_figures.py sstar        -> figures/fig_e13_sstar_vs_N.png       (stage 1 + stage 3 points; PBMC3k E7/E9 in grey)
    python e13_figures.py robustness   -> figures/fig_e13_robustness.png       (stage 2)
    python e13_figures.py paper        -> figures/fig_e13_paper_protocol.png   (stage 4)
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

E13 = Path(__file__).resolve().parents[1]
ROOT = E13.parent
RES, FIG = E13 / "results", E13 / "figures"
FIG.mkdir(exist_ok=True)
ONE_PASS = ("count_sketch", "hashing", "jl_sparse", "heavy_hitters_mg")
INK, INK2, GRID, GREY = "#1f1f1e", "#5f5e5a", "#e6e5e0", "#a9a8a2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]     # fixed categorical order
MARK = ["o", "s", "D", "^", "v", "P"]
SHORT = {"CD4+ T Helper2": "Th2 (0.14 %)", "CD34+": "CD34+ (0.40 %)", "CD4+/CD45RA+/CD25- Naive T": "naive CD4 T (2.7 %)",
         "CD19+ B": "B (8.6 %)", "Megakaryocytes": "Mk", "Dendritic cells": "DC", "B cells": "B"}
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def load(name):
    p = RES / name
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    assert d["schema_version"] == 1, name
    return d


def best(tk, key="S_star"):
    c = [(m.get(key), k) for k, m in tk["methods"].items() if k in ONE_PASS and m.get(key)]
    return min(c) if c else (None, None)


def fig_sstar():
    s1, s3 = load("e13_stage1.json"), load("e13_stage3.json")
    e9 = json.loads((ROOT / "HPC" / "results" / "e9_fullpairs.json").read_text())
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    grey = [(p["N"], p["best_S_star"]) for p in e9["e7_points"] + e9["e9_points"] if p.get("best_S_star")]
    ax.scatter([g[0] for g in grey], [g[1] for g in grey], s=26, color=GREY, marker="o", zorder=2, label="PBMC3k (E7 + E9), best one-pass")
    tasks = []
    for d in (s1, s3):
        if d:
            tasks += [t for t in d["tasks"] if t.get("split", 0) == 0 and "gate_vs_materialised" not in t and "methods" in t]
    order = list(dict.fromkeys(t["positive"] for t in tasks))
    for i, pos in enumerate(order):
        pts = sorted([(t["N"], best(t)[0], t.get("signal", True)) for t in tasks if t["positive"] == pos], key=lambda x: x[0])
        xs = [p[0] for p in pts if p[1]]
        ys = [p[1] for p in pts if p[1]]
        sig = all(p[2] for p in pts)
        lab = SHORT.get(pos, pos) + ("" if sig else " · no signal (exact BA < 0.6)")
        ax.plot(xs, ys, color=SERIES[i], lw=1.5, alpha=0.8, zorder=3)
        ax.scatter(xs, ys, s=44, color=SERIES[i] if sig else "white", edgecolors=SERIES[i], linewidths=1.6, marker=MARK[i], zorder=4,
                   label=lab)
    Nx = np.logspace(2.8, 8.6, 50)
    ax.plot(Nx, Nx, color=INK2, ls="--", lw=1)
    ax.text(3e7, 3e7 * 1.6, "S = N (store every feature)", color=INK2, fontsize=8, ha="right", rotation=0)
    ax.plot(Nx, np.ceil(np.log2(Nx)), color=INK, ls=":", lw=1.4)
    ax.text(1.5e4, 2.6, "dotted: log₂N index qubits (simulated quantum sketch; qubits and counters are different units)",
            color=INK, fontsize=7.5, va="top")
    ax.set_ylim(1.6, None)
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("feature-space size N (genes K=1024; gene pairs K=512, 2048, 8192, all ≈ 17,100 genes)")
    ax.set_ylabel("S* = counters the best one-pass sketch needs")
    ax.set_title("PBMC68k: classical memory S* needed to match the exact centroid\n"
                 "best of count-sketch / hashing / sparse JL / Misra–Gries · split 0 · stream at M_run (E7 protocol)",
                 fontsize=9.5, loc="left", color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_e13_sstar_vs_N.png")
    print("wrote fig_e13_sstar_vs_N.png")


def fig_robustness():
    d = load("e13_stage2.json")
    rob = d["robustness"]
    groups = [g for g in rob if g["n_splits"] > 0]
    fsets = sorted(dict.fromkeys(g["feature_set"] for g in groups), key=lambda f: min(g["N"] for g in groups if g["feature_set"] == f))
    fig, axes = plt.subplots(1, len(fsets), figsize=(4.2 * len(fsets), 4.4), sharey=True, squeeze=False)
    for ax, fs in zip(axes[0], fsets):
        gs = [g for g in groups if g["feature_set"] == fs]
        labels = []
        for i, g in enumerate(gs):
            ds = g["dataset"]
            c = INK if ds == "pbmc68k" else GREY
            v = np.array([x if x else np.nan for x in g["best_S_star_over_N_per_split"]], dtype=float)
            x = i
            ax.scatter(np.full(len(v), x), v, s=14, color=c, alpha=0.5, zorder=3)
            if np.isfinite(v).any():
                lm, ls = np.nanmean(np.log10(v)), np.nanstd(np.log10(v))
                ax.errorbar([x], [10**lm], yerr=[[10**lm - 10**(lm - ls)], [10**(lm + ls) - 10**lm]], fmt="_", color=c, ms=14, lw=1.6, zorder=4)
            labels.append(("68k " if ds == "pbmc68k" else "3k ") + SHORT.get(g["positive"], g["positive"]).split(" (")[0] + ("" if g["signal"] else "*"))
        ax.set_xticks(range(len(gs)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
        ax.set_yscale("log")
        ax.set_title(f"{fs} (N = {gs[0]['N']:,})", fontsize=9.5, loc="left")
    axes[0][0].set_ylabel("best one-pass S*/N\n(dots = splits; bar = geo. mean ± 1 sd)")
    fig.suptitle("Robustness: S*/N over 5 train/test splits × 5 seeds · black = PBMC68k, grey = PBMC3k (E7 setup) · * = no signal",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "fig_e13_robustness.png")
    print("wrote fig_e13_robustness.png")


def fig_paper():
    d = load("e13_stage4.json")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    show = [("jl", "paper: sparse JL s=1 + ridge"), ("bucket", "paper: feature hashing (unsigned) + ridge"),
            ("trunc", "paper: keep most-detected genes + ridge"), ("jl_tuned", "sparse JL s=1 + ridge, alpha tuned (nested)"),
            ("cs_select", "count-sketch gene selection + ridge"), ("cs_centroid", "one-pass count-sketch centroid")]
    for i, (m, lab) in enumerate(show):
        rows = d["curves"].get(m)
        if not rows:
            continue
        ax.errorbar([r["S"] for r in rows], [r["accuracy_mean"] for r in rows], yerr=[r["accuracy_sem"] for r in rows],
                    color=SERIES[i], marker=MARK[i], ms=4, lw=1.5, capsize=0, label=lab)
    fr = d["full_ridge"]["mean"]
    ax.axhline(fr, color=INK, ls=":", lw=1.2)
    ax.text(9, fr + 0.004, f"ridge on all {d['D']:,} genes = {fr:.3f} (ideal quantum LS-SVM accuracy)", fontsize=7.5, color=INK)
    ec = d["exact_centroid"]["accuracy"]["mean"]
    ax.axhline(ec, color=INK2, ls=":", lw=1)
    ax.text(9, ec + 0.004, f"exact centroid, all genes = {ec:.3f}", fontsize=7.5, color=INK2)
    q = d["quantum"]["paper_machine_size_mean"]
    ax.axvline(q, color=INK2, ls="--", lw=1)
    ax.text(q * 1.08, 0.53, f"quantum machine size\n(paper formula) ≈ {q:.0f}\nformula only, not run", fontsize=7.5, color=INK2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("classical machine size S (counters / features kept)")
    ax.set_ylabel("5-fold CV accuracy (paper's metric)")
    ax.set_title(f"Paper protocol (Fig. 2b style), PBMC68k: {d['n_pairs']} unique cell-type pairs (paper's {d['n_draws']} draws), D = {d['D']:,} genes\n"
                 "mean ± s.e.m. over pairs, weighted by draw count · RidgeClassifier(alpha=200, balanced) unless noted", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_ylim(0.45, 0.95)
    fig.tight_layout()
    fig.savefig(FIG / "fig_e13_paper_protocol.png")
    print("wrote fig_e13_paper_protocol.png")


if __name__ == "__main__":
    {"sstar": fig_sstar, "robustness": fig_robustness, "paper": fig_paper}[sys.argv[1]]()
