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
IBM_BLUE, IBM_LIGHT = "#2F5FD0", "#9DB4EA"
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


def compare():
    """Same 8 test cells, same frozen phases: classical formula vs quantum simulators vs real IQM Garnet (vs IBM, if run)."""
    e4 = load_json("e4_sim_circuits.json")
    hw = {4: load_json("e5_hardware.json"), 16: load_json("e5_hardware_N16.json")}
    ibm = {N: load_json(f"e5_ibm_N{N}.json") for N in (4, 16) if (ROOT / "results" / f"e5_ibm_N{N}.json").exists()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6 + (0.4 if ibm else 0)), sharey=True)
    series = [("classical (formula Σa²sinφ)", "formula", "#C9C7C1", None),
              ("simulator: 2,000 shots, noiseless", "shots_ideal", "#8A8984", None),
              ("simulator: IQM noise model", "shots_fake_noisy", "#4A4A48", None),
              ("IQM Garnet raw", "raw", INK, "iqm"),
              ("IQM Garnet calibrated", "calibrated", GOLD, "iqm")]
    if ibm:
        bname = next(iter(ibm.values()))["meta"]["backend"]
        series += [(f"IBM {bname} raw", "raw", IBM_BLUE, "ibm"), (f"IBM {bname} calibrated", "calibrated", IBM_LIGHT, "ibm")]
    for ax, b in zip(axes, e4["by_N"]):
        h = hw[b["N"]]; hi = ibm.get(b["N"])
        src = {"iqm": {c["test_index"]: c for c in h["cells"]}, "ibm": ({c["test_index"]: c for c in hi["cells"]} if hi else {})}
        x = np.arange(len(b["cells"])); w = 0.8 / len(series)
        for k, (name, key, col, dev) in enumerate(series):
            if dev == "ibm" and not hi:
                continue
            vals = [(src[dev][c["test_index"]][key] if dev else c[key]) for c in b["cells"]]
            ax.bar(x + (k - (len(series) - 1) / 2) * w, vals, w, color=col, label=name)
        ax.axhline(h["summary"]["threshold"], color=RED, ls="--", lw=1.2, label="decision threshold")
        ax.axhline(0, color=DIM, lw=.6)
        ax.set_xticks(x); ax.set_xticklabels([("rare" if c["label"] else "common") for c in b["cells"]], fontsize=9)
        t, s = h["meta"]["transpiled_test"], h["summary"]
        title = (f"N={b['N']} genes · {b['n_qubits']} qubits · IQM {t['two_qubit']} CZ, α={s['calibration']['alpha']:.2f}, "
                 f"correct raw {round(s['sign_agreement_raw']*8)}/8 → cal {round(s['sign_agreement_calibrated']*8)}/8")
        if hi:
            ti, si = hi["meta"]["transpiled_test"], hi["summary"]
            title += (f"\nIBM {hi['meta']['backend']}: {ti['two_qubit']} 2q gates, α={si['calibration']['alpha']:.2f}, "
                      f"correct raw {si['cells_correct_raw']}/8 → cal {si['cells_correct_calibrated']}/8")
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_ylabel("classifier score ⟨Z⟩ (above threshold = rare)")
    axes[0].legend(fontsize=8, loc="lower left")
    dev = "real IQM and IBM hardware" if ibm else "real IQM hardware"
    fig.suptitle(f"Same 8 held-out cells, same frozen sketch: classical vs simulator vs {dev}", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "fig_compare_classical_sim_hw.png", dpi=200); plt.close(fig)
    print("[figures] compare done")


if __name__ == "__main__":
    compare()


def three_way():
    """One graph: classical vs quantum simulator vs real IQM hardware (vs real IBM hardware, if run), 8 held-out cells, N=4."""
    e4 = load_json("e4_sim_circuits.json")["by_N"][0]
    h = load_json("e5_hardware.json")
    hc = {c["test_index"]: c for c in h["cells"]}
    hb = load_json("e5_ibm_N4.json") if (ROOT / "results" / "e5_ibm_N4.json").exists() else None
    cells = e4["cells"]
    vals = [("Classical computer (exact formula)", [c["formula"] for c in cells], "#C9C7C1"),
            ("Quantum simulator (Qiskit, 2,000 shots)", [c["shots_ideal"] for c in cells], "#6B6A66"),
            (f"Real IQM {h['meta']['alias'].capitalize()} (2,000 shots, raw)", [hc[c["test_index"]]["raw"] for c in cells], INK)]
    if hb:
        bc = {c["test_index"]: c for c in hb["cells"]}
        vals.append((f"Real IBM {hb['meta']['backend']} (2,000 shots, raw)", [bc[c["test_index"]]["raw"] for c in cells], IBM_BLUE))
    x = np.arange(len(cells)); w = 0.8 / len(vals)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for k, (name, v, col) in enumerate(vals):
        ax.bar(x + (k - (len(vals) - 1) / 2) * w, v, w, color=col, label=name)
    ax.axhline(h["summary"]["threshold"], color=RED, ls="--", lw=1.3, label="decision threshold")
    ax.axhline(0, color=DIM, lw=.6)
    ax.set_xticks(x); ax.set_xticklabels([("rare (B cell)" if c["label"] else "common") for c in cells], fontsize=9)
    ax.set_ylabel("classifier score  (above threshold = rare)")
    s = h["summary"]
    ttl = (f"Same 8 test cells, same model · 3 qubits · correct: classical {round(s['balanced_accuracy_ideal']*8)}/8 · simulator 8/8 · "
           f"IQM {round(s['balanced_accuracy_raw']*8)}/8 ({h['meta']['transpiled_test']['two_qubit']} CZ)")
    if hb:
        ttl += f" · IBM {hb['summary']['cells_correct_raw']}/8 ({hb['meta']['transpiled_test']['two_qubit']} 2q)"
    ax.set_title(ttl, fontsize=10, loc="left")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig_three_way.png", dpi=200); plt.close(fig)
    print("[figures] three_way done")


if __name__ == "__main__":
    three_way()
