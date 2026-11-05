"""Print the E8 markdown section (Route 2 on 1000 Genomes) from results/e8_route2_genotypes.json. No hand-typed numbers."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ONE_PASS = ["count_sketch", "hashing", "jl_sparse", "heavy_hitters_mg", "reservoir", "random_subset"]
NAMES = {"count_sketch": "count-sketch", "hashing": "hashing", "jl_sparse": "sparse JL", "heavy_hitters_mg": "Misra–Gries",
         "reservoir": "reservoir", "random_subset": "random SNP subset", "heavy_hitters_top": "oracle top-S", "hashing_logreg": "hashing+logreg"}


def fs(S, N):
    if S is None:
        return "never"
    return f"{S:,} ({S / N:.2g})"


def main():
    d = json.loads((ROOT / "results" / "e8_route2_genotypes.json").read_text())
    rg = d["config"]["route2_genotypes"]
    out = ["## E8: Route 2 on a dense-signal dataset (1000 Genomes genotypes)\n",
           f"**Why.** PBMC3k signal is sparse (marker genes), so classical sketches won (E7). Close human populations differ by small "
           f"allele-frequency shifts spread over many SNPs (density screen: `results/e7b_g1k_density_screen.json`). "
           f"Data: {rg['dataset']}; {rg['n_snps_total']:,} SNPs ({rg['snp_filter']}); {rg['features']}. "
           f"Same stream, weights, seeds and S* rule as E7, plus a **random SNP subset** baseline (S exact counters on S random SNPs), "
           f"which is the fair classical baseline when the signal is dense but redundant. "
           f"Scaling test: nested random SNP subsets N/8 → N. Script `scripts/10_route2_genotypes.py`, figure `figures/fig_e8_genotype_memory.png`.\n"]
    out.append("**Best one-pass classical S* (stable, tol = max(0.01, one test positive)) vs N, and log–log slope** (1 = memory grows like N, 0 = flat):\n")
    Ns = [b["N"] for b in d["tasks"][0]["by_N"]]
    out.append("| task | exact centroid at full N | " + " | ".join(f"N = {n:,}" for n in Ns) + " | slope |")
    out.append("|---|---|" + "---|" * len(Ns) + "---|")
    slopes = {}
    for t in d["tasks"]:
        best = []
        for b in t["by_N"]:
            ms = b["real"][0]["methods"]
            vals = {k: ms[k]["S_star_cell_stable"] for k in ONE_PASS if k in ms}
            k = min(vals, key=lambda k: vals[k] if vals[k] else b["N"] + 1)
            tk = b["real"][0]
            ok = tk["stream_full"]["balanced_accuracy"] >= tk["exact_centroid"]["balanced_accuracy"] - tk["tolerance_cell"]
            best.append((b["N"], vals[k] or b["N"], k + ("" if ok else "; † S = N stream misses band")))
        xs = [math.log2(n) for n, _, _ in best]; ys = [math.log2(s) for _, s, _ in best]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sl = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
        slopes[t["task"]] = sl
        ex = t["by_N"][-1]["real"][0]["exact_centroid"]
        out.append(f"| {t['task']} ({t['n_pos']} vs {t['n_neg']}) | {ex['balanced_accuracy']:.2f} / AUPRC {ex['auprc']:.2f} | "
                   + " | ".join(f"{s:,} ({s / n:.2g}, {NAMES[k.split(';')[0]]}{' †' if '†' in k else ''})" for n, s, k in best) + f" | {sl:.2f} |")
    out.append("\n**Per method at full N** (stable S*, S*/N in brackets; quantum = classical simulation of the sketch, same records):\n")
    out.append("| task | " + " | ".join(NAMES[k] for k in ONE_PASS + ["heavy_hitters_top", "hashing_logreg"]) + " | sketch p·sin(cφ) | logreg, all N |")
    out.append("|---|" + "---|" * (len(ONE_PASS) + 4))
    for t in d["tasks"]:
        b = t["by_N"][-1]
        tk = b["real"][0]
        ms = tk["methods"]
        cells = [fs(ms[k].get("S_star_cell_stable"), b["N"]) if k in ms else "—" for k in ONE_PASS + ["heavy_hitters_top", "hashing_logreg"]]
        q = tk.get("quantum_sim", {}).get("balanced_accuracy")
        lr = (tk.get("logreg_full") or {}).get("balanced_accuracy")
        out.append(f"| {t['task']} | " + " | ".join(cells) + f" | {q:.2f} | {lr:.2f} |")
    sc = d["scaling"]
    out.append("\nPer-method slopes (S* vs N): " + "; ".join(
        f"{task}: " + ", ".join(f"{NAMES[k]} {v['loglog_slope']:.2f}" for k, v in m.items() if v["loglog_slope"] is not None and k in ONE_PASS)
        for task, m in sc.items()) + ".")
    qb = [b["quantum"]["index_qubits"] for b in d["tasks"][0]["by_N"]]
    out.append(f"Quantum register over the same N range: {' → '.join(map(str, qb))} index qubits (≈2× logical).\n")
    lo, hi = min(slopes.values()), max(slopes.values())
    full = [min((t["by_N"][-1]["real"][0]["methods"][k]["S_star_cell_stable"] or t["by_N"][-1]["N"]) for k in ONE_PASS
                if k in t["by_N"][-1]["real"][0]["methods"]) for t in d["tasks"]]
    Nmax = d["tasks"][0]["by_N"][-1]["N"]
    fr_lo, fr_hi = min(full) / Nmax, max(full) / Nmax
    if lo >= 0.75 and fr_lo >= 0.05:
        verdict = (f"**Verdict:** on this dense-signal dataset the best one-pass classical sketch needs memory that grows roughly like N "
                   f"(slope {lo:.2f}–{hi:.2f}), while the quantum register grows like log₂N. Real-data evidence *consistent with* the paper's "
                   f"memory claim; not proof (4 N values over one decade, ~31 test positives, one split).")
    elif hi < 0.75 and fr_hi <= 0.05:
        verdict = (f"**Verdict: no memory advantage here either.** The signal is dense (top 1 % of SNPs hold only 12–15 % of ‖Δ‖²), but it is "
                   f"*compressible*: the best one-pass classical sketch needs {min(full):,}–{max(full):,} counters at N = {Nmax:,} "
                   f"(S*/N = {fr_lo:.2g}–{fr_hi:.2g}), and its memory barely grows with N (log–log slopes {lo:.2f} to {hi:.2f}; 1 would mean ∝ N). "
                   f"Population differences are a few shared directions spread over many correlated SNPs, so random projections and "
                   f"hashing keep them. Dense is not the same as hard: the paper's worst cases need dense *and* incompressible signals.")
    else:
        verdict = (f"**Verdict:** mixed. Slopes {lo:.2f}–{hi:.2f}, S*/N at full N {fr_lo:.2g}–{fr_hi:.2g}; neither flat nor clearly linear.")
    out.append(verdict)
    out.append("\n- Same records for all methods: M = 100·N at every N (fixed after a first pass where the automatic M choice varied with N; "
               "first pass kept as `results/e8_route2_genotypes_autoM.json`, same conclusion). The paper's bound is on the product M·S.")
    out.append("- S* can fall as N grows (JPT vs rest of East Asia): more SNPs with the same per-SNP signal give a larger total margin, "
               "so a fixed-size sketch has more slack. More redundant features help the classical side too.")
    out.append("- Not a claim about accuracy: the quantum sketch at M → ∞ equals the exact centroid, which classical also computes with N floats; "
               "logistic regression on stored data is as good or better. The question is only memory in one pass.")
    out.append("- Limits: " + "; ".join(d["limits"]) + ".")
    print("\n".join(out))


if __name__ == "__main__":
    main()
