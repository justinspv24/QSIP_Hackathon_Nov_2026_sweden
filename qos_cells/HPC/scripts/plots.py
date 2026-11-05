"""All HPC figures, built from HPC/results/*.json only (no data, no GPU).

    python plots.py            # every figure whose JSON exists
    python plots.py e9 e10     # selected
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpc_common as hc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, GOLD, DIM, RED, BLUE = "#08090B", "#D9A800", "#8A8984", "#D94F3D", "#2F6DB5"
STYLE = {"count_sketch": (INK, "o", "count-sketch"), "hashing": ("#4A4A48", "s", "feature hashing"),
         "jl_sparse": (DIM, "^", "sparse JL"), "heavy_hitters_mg": (GOLD, "D", "Misra–Gries"),
         "heavy_hitters_top": ("#B08900", "d", "oracle top-S (bound)"), "reservoir": (RED, "x", "reservoir"),
         "hashing_logreg": (BLUE, "p", "hashing + logreg (not one-pass)")}
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
                     "axes.titlesize": 15, "legend.fontsize": 11.5})
W, H = 13.33, 7.5


def e9():
    d = hc.load_result("e9_fullpairs.json")
    classes = ["B cells", "Dendritic cells", "Megakaryocytes"]
    fig, axes = plt.subplots(1, 3, figsize=(W, H), sharey=True)
    pts = {}
    for blk in d["by_K"]:
        for t in blk["real"]:
            if "methods" in t:
                pts.setdefault(t["positive"], []).append((blk["N"], blk["K"], t))
    Nall = sorted({blk["N"] for blk in d["by_K"]} | {p["N"] for p in d["e7_points"]})
    xs = np.array([min(Nall) / 2, max(Nall) * 2])
    for ax, c in zip(axes, classes):
        rows = sorted(pts.get(c, []), key=lambda r: r[0])
        for key, (col, mk, lab) in STYLE.items():
            X, Y, Xn = [], [], []
            for N, K, t in rows:
                m = t["methods"].get(key)
                if m is None:
                    continue
                v = m.get("S_star_cell_stable")
                if v:
                    X.append(N)
                    Y.append(v)
                else:
                    Xn.append(N)
            ls = "--" if key in ("heavy_hitters_top", "hashing_logreg") else "-"
            if X:
                ax.plot(X, Y, ls, marker=mk, color=col, lw=1.6, ms=7, label=lab, alpha=0.9)
            if Xn:
                ax.plot(Xn, [max(Nall) * 1.6] * len(Xn), marker=mk, color=col, ls="none", ms=8, alpha=0.6)
        # E7 published values (best one-pass) as hollow stars
        e7 = [p for p in d["e7_points"] if p["positive"] == c and p["best_S_star"]]
        if e7:
            ax.plot([p["N"] for p in e7], [p["best_S_star"] for p in e7], "*", mfc="none", mec=RED, ms=16, mew=1.6,
                    label="E7 best one-pass (published)")
        ax.plot(xs, xs, color=INK, lw=0.9, ls=":")
        ax.text(3e7, 3e7 * 1.8, "S = N", fontsize=11, rotation=0)
        ax.plot(xs, np.log2(xs), color=RED, lw=2.4, label="quantum sketch: log₂N index qubits")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xs)
        ax.set_ylim(4, max(Nall) * 4)
        ax.axhline(max(Nall) * 1.6, color=DIM, lw=0.6, ls=":")
        npos = rows[0][2]["n_test_pos"] if rows else "?"
        ax.set_title(f"{c}  ({npos} test positives)", loc="left")
        ax.set_xlabel("feature dimension N (gene pairs)")
        for N, K, _ in rows:
            ax.text(N, 5.2, f"K={K}", fontsize=9.5, ha="center", va="bottom", color=DIM, rotation=90)
    axes[0].set_ylabel("S* = counters needed (log)")
    axes[0].text(min(Nall) / 1.8, max(Nall) * 1.6, "never", fontsize=10, color=DIM, va="center")
    H_, L_ = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in L_:
                H_.append(h)
                L_.append(l)
    fig.legend(H_, L_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("E9 · PBMC3k gene pairs up to N = 9.4·10⁷: classical sketches need only S* ≈ 8–4,096 counters; "
                 "S* does not grow with N\nS* = smallest S whose mean balanced accuracy stays within max(0.01, one test cell) "
                 "of the exact centroid (stable rule, as E7)", x=0.01, ha="left", fontsize=14.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.25, wspace=0.08)
    out = hc.FIGURES / "fig_e9_sstar_vs_N.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def e10():
    d = hc.load_result("e10_atlas.json")
    tasks = [t for t in d["tasks"] if "methods" in t]
    pairs = sorted([t for t in tasks if t["space"] == "pairs"], key=lambda t: t["pi_hat"])
    genes = {t["task"]: t for t in tasks if t["space"] == "genes"}
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, len(pairs) + 1, height_ratios=[1, 1], width_ratios=[1] * len(pairs) + [1.15])
    for i, t in enumerate(pairs):
        ax = fig.add_subplot(gs[0, i])
        ex = t["exact_centroid"]["balanced_accuracy"]
        for key, (col, mk, lab) in STYLE.items():
            m = t["methods"].get(key)
            if not m:
                continue
            S = [r["S"] for r in m["rows"]]
            y = [r["balanced_accuracy_mean"] for r in m["rows"]]
            ax.plot(S, y, ls="--" if key in ("heavy_hitters_top", "hashing_logreg") else "-", marker=mk, ms=3.5, lw=1.2, color=col, label=lab)
        ax.axhline(ex, color=INK, ls=":", lw=1.2)
        tol = t.get("tolerance_cell", 0.01)
        ax.axhspan(ex - tol, ex + tol, color=INK, alpha=0.06, lw=0)
        if "logreg_full" in t:
            ax.axhline(t["logreg_full"]["balanced_accuracy"], color=BLUE, ls="-.", lw=1)
        ax.axvline(t["N"], color=INK, ls="--", lw=0.8)
        ax.axvline(t["index_qubits"], color=RED, ls="--", lw=1.2)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0.45, 1.02)
        short = {"CD1c-positive myeloid dendritic cell": "CD1c+ DC", "plasma cell": "plasma cell",
                 "hematopoietic stem cell": "HSC", "CD14-positive monocyte": "CD14+ mono"}.get(t["task"], t["task"])
        ax.set_title(f"{short}\nπ={t['pi_hat']:.1e} · {t['n_test_pos']} test pos", fontsize=12, loc="left")
        ax.set_xlabel("memory S (counters)", fontsize=11)
        ax.tick_params(labelsize=10)
        if i == 0:
            ax.set_ylabel("balanced acc. (test)", fontsize=12)
    # sample cost panel
    ax = fig.add_subplot(gs[:, -1])
    for space, mk, col in (("pairs", "o", INK), ("genes", "s", GOLD)):
        ts = [t for t in tasks if t["space"] == space and t.get("sample_cost")]
        if not ts:
            continue
        x = np.array([1 / (t["pi_hat"] * (1 - t["pi_hat"])) for t in ts])
        y = np.array([t["sample_cost"]["M10_measured"] / t["N"] for t in ts])
        yc = np.array([t["sample_cost"]["M10_closed_form"] / t["N"] for t in ts])
        ax.plot(x, y, mk, color=col, ms=9, label=f"measured ({space})")
        ax.plot(x, yc, mk, mfc="none", mec=col, ms=14, mew=1.4, label=f"closed form ({space})")
        kap = np.median([t["sample_cost"]["law_constant_kappa"] / t["N"] for t in ts])
        xx = np.logspace(np.log10(x.min() / 2), np.log10(x.max() * 2), 20)
        ax.plot(xx, 100 * kap * xx, color=col, lw=1, ls=":", label=f"100·κ̃/(π(1−π)), {space}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1 / (π(1−π))")
    ax.set_ylabel("records for 10 % error,  M₁₀ / N")
    ax.set_title("rare classes cost records, not memory", fontsize=13, loc="left")
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    # bottom row: summary table as text
    axt = fig.add_subplot(gs[1, :-1])
    axt.axis("off")
    hdr = ["task", "π", "test+", "exact", "logreg", "stream", "q-sim", "best one-pass S*", "d_eff"]
    lines = []
    for t in pairs:
        b = t.get("best_one_pass") or {}
        g = genes.get(t["task"])
        lines.append([t["task"][:28], f"{t['pi_hat']:.1e}", str(t["n_test_pos"]),
                      f"{t['exact_centroid']['balanced_accuracy']:.3f}", f"{t.get('logreg_full', {}).get('balanced_accuracy', float('nan')):.3f}",
                      f"{t['stream_full']['balanced_accuracy']:.3f}", f"{t['quantum_sim']['balanced_accuracy']:.3f}",
                      f"{b.get('S_star')} ({b.get('method', '-')})", f"{t['d_eff']:.0f}"])
        if g:
            bg = g.get("best_one_pass") or {}
            lines.append(["   genes only (N=512)", "", "", f"{g['exact_centroid']['balanced_accuracy']:.3f}",
                          f"{g.get('logreg_full', {}).get('balanced_accuracy', float('nan')):.3f}", f"{g['stream_full']['balanced_accuracy']:.3f}",
                          f"{g['quantum_sim']['balanced_accuracy']:.3f}", f"{bg.get('S_star')} ({bg.get('method', '-')})", f"{g['d_eff']:.0f}"])
    tb = axt.table(cellText=lines, colLabels=hdr, loc="center", cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10.5)
    tb.scale(1, 1.35)
    for (r, c), cell in tb.get_celld().items():
        cell.set_linewidth(0.3)
        if r == 0:
            cell.set_text_props(weight="bold")
    ds = d["dataset"]
    fig.suptitle(f"E10 · AIDA atlas ({ds['n_cells']:,} PBMCs, {ds['n_donors']} donors, curated labels, donor split): "
                 "gene-pair task (N = 131,328) is solved by small classical sketches too", x=0.01, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = hc.FIGURES / "fig_e10_atlas.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def external(script):
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / script), "figure"], check=False)


if __name__ == "__main__":
    which = sys.argv[1:] or ["e8", "e9", "e10", "e11"]
    for w in which:
        name = {"e8": "e8_gpu_circuit.json", "e9": "e9_fullpairs.json", "e10": "e10_atlas.json", "e11": "e11_advantage_map.json"}[w]
        if not (hc.RESULTS / name).exists():
            print("skip", w, "(no JSON)")
            continue
        if w == "e9":
            e9()
        elif w == "e10":
            e10()
        elif w == "e8":
            external("e8_gpu_circuit.py")
        elif w == "e11":
            external("e11_advantage_map.py")
