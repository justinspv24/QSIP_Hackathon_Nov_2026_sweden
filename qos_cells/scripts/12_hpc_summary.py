"""Collect the HPC results (E8' to E14) into one small JSON for the dashboard.
Reads HPC/results, HPC_E13/results, HPC_E14/results. Nothing is typed by hand."""
import json, math, pathlib
R = pathlib.Path(__file__).resolve().parents[1]
L = lambda p: json.loads((R / p).read_text(encoding="utf-8"))
e8, e9, e10, e11 = (L(f"HPC/results/{f}") for f in ["e8_gpu_circuit.json", "e9_fullpairs.json", "e10_atlas.json", "e11_advantage_map.json"])
e13s3, e13s4, e14 = L("HPC_E13/results/e13_stage3.json"), L("HPC_E13/results/e13_stage4.json"), L("HPC_E14/results/e14_stage1.json")

scale, excluded = [], []
for p in e9["e7_points"] + e9["e9_points"]:
    scale.append({"dataset": "PBMC3k", "source": p["source"], "task": p["positive"], "N": p["N"], "S_star": p["best_S_star"]})
for r in e10["summary_table"]:
    if r["exact_ba"] < 0.65:
        excluded.append(f'{r["task"]} N={r["N"]} (exact bal. acc {r["exact_ba"]:.2f})'); continue
    if r.get("best_S_star") is not None:
        scale.append({"dataset": "AIDA atlas (1.27 M cells)", "source": "E10", "task": r["task"], "N": r["N"], "S_star": r["best_S_star"]})
for t in e13s3["tasks"]:
    b = t["best_one_pass"]
    if t["exact_centroid"]["balanced_accuracy"] < 0.65:   # no signal (e.g. Th2): S* is meaningless
        excluded.append(f'{t["positive"]} N={t["N"]} (exact bal. acc {t["exact_centroid"]["balanced_accuracy"]:.2f})'); continue
    if b.get("S_star") is not None:
        scale.append({"dataset": "PBMC68k", "source": "E13", "task": t["positive"], "N": t["N"], "S_star": b["S_star"]})

def curve(rows, key="S"):
    return [{"S": r[key], "acc": r["accuracy_mean"]} for r in rows]
paper = {k: curve(e13s4["curves"][k]) for k in ["jl", "bucket"]}
e14c = {k: curve(e14["curves"][k]["a"]) for k in ["m1_log_1p", "m1_sq_1p", "m0_hashridge", "m2_select", "m3_fd"] if k in e14["curves"]}
labels = {**{k: e13s4["labels"][k] for k in paper}, **{k: e14["labels"][k] for k in e14c}}

out = {
    "schema_version": 1, "experiment": "e9_e14_summary",
    "e8": {"max_abs_diff": e8["summary"]["max_abs_circuit_minus_formula_all_sizes"], "largest_n": e8["summary"]["largest_n"]},
    "e10": {k: e10["dataset"][k] for k in ["n_cells", "n_donors", "n_genes"]},
    "e11": {"caption": e11["caption"], "real_points": [{"label": p["label"], "log2N": p["log2N"], "y": p["y"], "S_over_N": p["best_S_star_over_N"]} for p in e11["real_points"]]},
    "scale": scale, "excluded_no_signal": excluded,
    "gap": {"full_ridge": e13s4["full_ridge"]["mean"], "full_ridge_nested": e14["full_ridge_reference"]["alpha_nested_acc"]["accuracy_mean"],
            "quantum_size": e13s4["quantum"]["paper_machine_size_mean"], "curves": {**paper, **e14c}, "labels": labels,
            "S_star": e14["S_star_within_001_of_full_ridge"]},
}
(R / "results/e9_e14_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(len(scale), "scale points;", list(out["gap"]["curves"]))
