"""All figures from results/*.json (never from hand-typed numbers). Writes figures/*.png."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qos_cells import ROOT
from qos_cells.io import load_json

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
INK, GOLD, DIM, RED = "#08090B", "#D9A800", "#8A8984", "#D94F3D"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def e1():
    d = load_json("e1_baselines.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = {"B cells": INK, "Dendritic cells": GOLD, "Megakaryocytes": RED}
    for t in d["tasks"]:
        Ns = [r["N"] for r in t["by_N"]]
        for ax, key in zip(axes, ["balanced_accuracy", "auprc"]):
            ax.plot(Ns, [r["exact_centroid"][key] for r in t["by_N"]], "o-", color=cols[t["positive"]], label=f"{t['positive']} (π={t['base_rate']:.3f})")
            ax.plot(Ns, [r["logreg_p"][key] for r in t["by_N"]], "s--", color=cols[t["positive"]], alpha=.45, ms=4)
            ax.axhline(t["logreg_all_genes"][key], color=cols[t["positive"]], lw=.8, ls=":", alpha=.6)
    for ax, lab in zip(axes, ["balanced accuracy", "AUPRC"]):
        ax.set_xscale("log", base=2); ax.set_xlabel("genes N (register = log₂N qubits)"); ax.set_ylabel(lab); ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_title("solid: exact centroid (M→∞ sketch) · dashed: logistic regression on p · dotted: logreg on all genes", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig_e1_baselines.png"); plt.close(fig)


def e2():
    d = load_json("e2_scaling.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    shades = ["#C9C7C1", "#8A8984", "#4A4A48", INK]
    for c, col in zip(d["curves"], shades):
        M = [r["M"] for r in c["rows"]]
        ax.errorbar(M, [r["rel_error_mean"] for r in c["rows"]], yerr=[r["rel_error_std"] for r in c["rows"]], fmt="o-", color=col, label=f"N={c['N']}", ms=4)
        ax.plot(M, [r["rel_error_predicted"] for r in c["rows"]], "--", color=col, alpha=.7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("records streamed M"); ax.set_ylabel("relative error of the sketch ‖φ−φ*‖/‖φ*‖")
    ax.set_title("solid: measured (5 seeds) · dashed: closed-form prediction, no fit", fontsize=8, loc="left"); ax.legend(fontsize=8)
    ax = axes[1]
    cl = d["classifier"]
    M = [r["M"] for r in cl["rows"]]
    ax.errorbar(M, [r["balanced_accuracy_mean"] for r in cl["rows"]], yerr=[r["balanced_accuracy_std"] for r in cl["rows"]], fmt="o-", color=INK, label="balanced accuracy", ms=4)
    ax.errorbar(M, [r["auprc_mean"] for r in cl["rows"]], yerr=[r["auprc_std"] for r in cl["rows"]], fmt="s-", color=GOLD, label="AUPRC", ms=4)
    ax.axhline(cl["exact_centroid"]["balanced_accuracy"], color=INK, ls=":", lw=1); ax.axhline(cl["exact_centroid"]["auprc"], color=GOLD, ls=":", lw=1)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.set_xlabel("records streamed M"); ax.set_ylabel("test score")
    ax.set_title(f"classifier at N={cl['N']} ({int(np.log2(cl['N']))} qubits), B cells · dotted: M→∞", fontsize=8, loc="left"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig_e2_scaling.png"); plt.close(fig)


def e3():
    d = load_json("e3_imbalance.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    shades = [INK, "#4A4A48", GOLD, RED]
    for s, col in zip(d["synthetic"], shades):
        M = [r["M"] for r in s["rows"]]
        ax.errorbar(M, [r["rel_error_mean"] for r in s["rows"]], yerr=[r["rel_error_std"] for r in s["rows"]], fmt="o-", color=col, ms=4, label=f"π={s['pi']}")
        ax.plot(M, [r["rel_error_predicted"] for r in s["rows"]], "--", color=col, alpha=.7)
    ax.axhline(0.1, color=DIM, lw=.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("records streamed M"); ax.set_ylabel("relative error of the sketch")
    ax.set_title(f"synthetic, N={d['N']}: only the class fraction π changes", fontsize=8, loc="left"); ax.legend(fontsize=8)
    ax = axes[1]
    pis = [s["pi"] for s in d["synthetic"]]
    x = np.arange(len(pis))
    ax.bar(x - .2, [s["cost_ratio_vs_balanced_measured"] for s in d["synthetic"]], .4, color=INK, label="measured: M(err=0.1) / M(π=0.5)")
    ax.bar(x + .2, [s["cost_ratio_vs_balanced_law"] for s in d["synthetic"]], .4, color=GOLD, label="law: [1/π(1−π)] / 4")
    ax.set_xticks(x); ax.set_xticklabels([f"π={p}" for p in pis]); ax.set_yscale("log"); ax.set_ylabel("sample cost relative to balanced")
    ax.set_title("rare classes cost samples, not qubits", fontsize=8, loc="left"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_e3_imbalance.png"); plt.close(fig)


def e5():
    runs = [("e5_hardware.json", "N=4 · 3 qubits"), ("e5_hardware_N16.json", "N=16 · 5 qubits")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (name, title) in zip(axes, runs):
        try:
            d = load_json(name)
        except FileNotFoundError:
            ax.text(.5, .5, "pending", ha="center"); continue
        cells = d["cells"]; x = np.arange(len(cells)); w = .27
        ax.bar(x - w, [c["ideal"] for c in cells], w, color="#C9C7C1", label="ideal (simulator)")
        ax.bar(x, [c["raw"] for c in cells], w, color=INK, label=f"IQM {d['meta']['alias']} raw")
        ax.bar(x + w, [c["calibrated"] for c in cells], w, color=GOLD, label="IQM calibrated")
        ax.axhline(d["summary"]["threshold"], color=RED, lw=1, ls="--", label="decision threshold")
        ax.set_xticks(x); ax.set_xticklabels([("rare" if c["label"] else "common") for c in cells], fontsize=8, rotation=0)
        s, t = d["summary"], d["meta"]["transpiled_test"]
        ax.set_title(f"{title} · {t['two_qubit']} CZ, depth {t['depth']} · α={s['calibration']['alpha']:.2f}\nsign agreement raw {s['sign_agreement_raw']:.0%} → calibrated {s['sign_agreement_calibrated']:.0%}", fontsize=8, loc="left")
        ax.set_ylabel("Hadamard-test score ⟨Z⟩"); ax.set_ylim(-1.05, 0.45)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig_e5_hardware.png"); plt.close(fig)


if __name__ == "__main__":
    for f in (e1, e2, e3, e5):
        f(); print(f"[figures] {f.__name__} done")
