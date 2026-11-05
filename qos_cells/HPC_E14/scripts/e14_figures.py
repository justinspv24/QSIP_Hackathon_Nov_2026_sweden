"""E14 figures, from JSON only (results/e14_stage1.json + HPC_E13/results/e13_stage4.json).
fig_e14_gap.png       accuracy (paper metric) vs memory: left = paper accounting (a), right = true peak floats (b)
fig_e14_gap_bacc.png  same, methods tuned on and scored by balanced accuracy
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

E14 = Path(__file__).resolve().parents[1]
E13J = E14.parent / "HPC_E13" / "results" / "e13_stage4.json"

E13_TRUE = {   # true peak floats of the E13 Stage-4 learners at size S (one pass, sufficient statistics)
    "jl": lambda S: S * (S + 1) / 2 + 2 * S + 3,
    "bucket": lambda S: S * (S + 1) / 2 + 2 * S + 3,
    "trunc": lambda S: S * (S + 1) / 2 + 3 * S + 3,       # + S gene ids; excludes the D gene-frequency counters
    "cs_select": lambda S: max(3 * S, S * (S + 1) / 2 + 3 * S + 4),
    "cs_centroid": lambda S: S + 3,
}
E13_LAB = {"jl": "E13 signed hashing + ridge", "bucket": "E13 paper hashing + ridge", "trunc": "E13 rare-gene truncation + ridge",
           "cs_select": "E13 count-sketch selection + ridge", "cs_centroid": "E13 count-sketch centroid"}
STY = {
    "m1_sq_1p": dict(c="#1f77b4", m="o", ls="-", lab="M1 hashing + SGD (squared), 1 pass"),
    "m1_log_1p": dict(c="#17becf", m="o", ls="-", lab="M1 hashing + SGD (logistic), 1 pass"),
    "m1_sq_5ep": dict(c="#1f77b4", m="^", ls=":", lab="M1 squared, 5 epochs [multi-pass]"),
    "m1_log_5ep": dict(c="#17becf", m="^", ls=":", lab="M1 logistic, 5 epochs [multi-pass]"),
    "m2_select": dict(c="#d62728", m="s", ls="--", lab="M2/M4 sketch selection + exact ridge / RLS [2 passes]"),
    "m3_fd": dict(c="#2ca02c", m="D", ls="-", lab="M3 hashing + Frequent Directions ridge, 1 pass"),
    "m0_hashridge": dict(c="#9467bd", m="v", ls="-.", lab="control: hashing + exact ridge, same nested tuning"),
}


def main():
    d = json.loads((E14 / "results" / "e14_stage1.json").read_text())
    e13 = json.loads(E13J.read_text())
    ref = d.get("full_ridge_reference") or {}
    for suffix, crit, metric, fn in (("", "acc", "accuracy_mean", "fig_e14_gap.png"),
                                     ("__tuned_on_bacc", "bacc", "balanced_accuracy_mean", "fig_e14_gap_bacc.png")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
        for ax, reg in zip(axes, ("a", "b")):
            for k, f in E13_TRUE.items():
                rows = e13["curves"].get(k, [])
                xs = [r["S"] if reg == "a" else f(r["S"]) for r in rows]
                ax.plot(xs, [r[metric] for r in rows], color="0.65", lw=1.2, ls="-" if k != "cs_centroid" else "--",
                        label=E13_LAB[k] if k in ("jl", "trunc", "cs_select", "cs_centroid") else None, zorder=1)
            for name, st in STY.items():
                rows = d["curves"].get(name + suffix, {}).get(reg, [])
                if not rows:
                    continue
                xs = [r["mem_a_chosen" if reg == "a" else "mem_b_chosen"] for r in rows]
                ax.plot(xs, [r[metric] for r in rows], color=st["c"], marker=st["m"], ls=st["ls"], ms=5, lw=1.6,
                        mfc="white" if "multi" in st["lab"] or "2 passes" in st["lab"] else st["c"], label=st["lab"], zorder=3)
            full = e13["full_ridge"]["mean"] if crit == "acc" else e13["full_ridge"]["balanced"]["mean"]
            ax.axhline(full, color="k", lw=1.2, label=f"full ridge, all genes, alpha=200 (paper) {full:.3f}")
            tuned = (ref.get(f"alpha_nested_{crit}") or {}).get(metric)
            if tuned:
                ax.axhline(tuned, color="k", lw=1.0, ls="--", label=f"full ridge, alpha by same nested CV {tuned:.3f}")
            ax.axvspan(42, 44, color="orange", alpha=0.35, label="paper's quantum machine size (42-44, formula)")
            ax.set_xscale("log")
            ax.set_xlim(8, 1e8 if reg == "b" else 2e4)
            ax.grid(alpha=0.3, which="both")
            ax.set_xlabel("memory, paper accounting (a): model + adaptive state, Gram not counted" if reg == "a"
                          else "memory, true peak floats (b): everything kept, incl. Gram / sums / counters")
            ax.set_title(f"({reg}) " + ("paper's accounting" if reg == "a" else "true peak floats"))
        axes[0].set_ylabel(("CV accuracy (paper metric)" if crit == "acc" else "CV balanced accuracy")
                           + ("" if crit == "acc" else ", tuned on balanced accuracy"))
        axes[0].set_ylim(0.6 if crit == "acc" else 0.55, 0.93 if crit == "acc" else 0.9)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
        fig.suptitle("E14: PBMC68k, 46 paper pairs, 5-fold CV - stronger classical learners at fixed memory "
                     + ("(tuned on accuracy)" if crit == "acc" else "(tuned on balanced accuracy)"), fontsize=11)
        fig.tight_layout(rect=(0, 0.17, 1, 0.96))
        fig.savefig(E14 / "figures" / fn, dpi=150)
        print("wrote", fn)


if __name__ == "__main__":
    sys.exit(main())
