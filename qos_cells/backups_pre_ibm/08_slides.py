"""Build presentation/plan3.html from presentation/plan3_template.html + results/*.json.
The deck contains no hand-typed numbers: everything comes from the JSON files written by scripts 01–06 and 09."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qos_cells import ROOT
from qos_cells.io import load_json


def compact_e7():
    """Route 2 (E7): the subset of results/e7_route2.json the backup slide needs."""
    p = ROOT / "results" / "e7_route2.json"
    if not p.exists():
        return None
    e7 = load_json("e7_route2.json")

    def task(t):
        return dict(task=t["task"], positive=t.get("positive", "synthetic"), N=t["N"], pi_hat=t["pi_hat"], n_test_pos=t["n_test_pos"],
                    exact=t["exact_centroid"], logreg_full=t.get("logreg_full"), M_run_over_N=t["M_run_over_N"],
                    stream_full=t["stream_full"], quantum_sim=t.get("quantum_sim"),
                    tolerance_cell=t.get("tolerance_cell", 0.01),
                    methods={k: dict(label=m["label"], S_star=m["S_star"], S_star_over_N=m["S_star_over_N"], S_star_stable=m["S_star_stable"],
                                     S_star_cell_stable=m.get("S_star_cell_stable"), S_star_cell_stable_over_N=m.get("S_star_cell_stable_over_N"),
                                     bytes_at_S_star=m["bytes_at_S_star"], bytes_at_S_star_cell_stable=m.get("bytes_at_S_star_cell_stable"),
                                     rows=[dict(S=r["S"], ba=r["balanced_accuracy_mean"], sd=r["balanced_accuracy_std"]) for r in m["rows"]])
                             for k, m in t["methods"].items()})
    return dict(by_K=[dict(K=b["K"], N=b["N"], quantum=b["quantum"], real=[task(t) for t in b["real"]],
                           synthetic=task(b["synthetic"]) if b.get("synthetic") else None) for b in e7["by_K"]],
                verdict=e7["verdict"], limits=e7["limits"], partial=e7.get("partial", False),
                S_star_rule=e7["config"]["route2"]["S_star_rule"])


ONE_PASS = ["count_sketch", "hashing", "jl_sparse", "heavy_hitters_mg", "reservoir", "random_subset"]


def compact_e8():
    """E8 (Route 2 on 1000 Genomes): per task and N, exact centroid and each one-pass method's stable S*."""
    p = ROOT / "results" / "e8_route2_genotypes.json"
    if not p.exists():
        return None
    e8 = load_json("e8_route2_genotypes.json")
    tasks = []
    for t in e8["tasks"]:
        rows = []
        for b in t["by_N"]:
            tk = b["real"][0]
            ss = {k: tk["methods"][k].get("S_star_cell_stable") for k in ONE_PASS if k in tk["methods"]}
            fin = [v if v else b["N"] for v in ss.values()]
            best_k = min(ss, key=lambda k: ss[k] if ss[k] else b["N"] + 1)
            rows.append(dict(N=b["N"], qubits=b["quantum"]["index_qubits"], exact=tk["exact_centroid"]["balanced_accuracy"],
                             tol=tk["tolerance_cell"], logreg=(tk.get("logreg_full") or {}).get("balanced_accuracy"),
                             M_over_N=tk["M_run_over_N"], S_star=ss, best_method=best_k, best_S=min(fin),
                             n_test_pos=tk["n_test_pos"], pi_hat=tk["pi_hat"],
                             stream_ok=tk["stream_full"]["balanced_accuracy"] >= tk["exact_centroid"]["balanced_accuracy"] - tk["tolerance_cell"]))
        Ns = [r["N"] for r in rows]
        import math
        slope = None
        if len(rows) >= 2:
            xs = [math.log2(r["N"]) for r in rows]; ys = [math.log2(r["best_S"]) for r in rows]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
        tasks.append(dict(task=t["task"], n_pos=t["n_pos"], n_neg=t["n_neg"], rows=rows, best_slope=slope))
    return dict(tasks=tasks, n_snps=e8["config"]["route2_genotypes"]["n_snps_total"], chromosomes=e8["chromosomes"],
                partial=e8.get("partial", False), limits=e8.get("limits", []))


def compact():
    if not all((ROOT / "results" / f).exists() for f in ("e1_baselines.json", "e2_scaling.json", "e3_imbalance.json", "e4_sim_circuits.json",
                                                         "e5_hardware.json", "e5_hardware_N16.json", "e6_resources.json")):
        # E1–E6 result files absent (e.g. a rebuild away from the laptop): reuse the compact data of the previous build
        prev = json.loads((ROOT / "results" / "slides_data.json").read_text())
        prev["e7"] = compact_e7()
        prev["e8"] = compact_e8()
        prev["generated_from"] = sorted(set(prev.get("generated_from", [])) | {"e7_route2.json"} | ({"e8_route2_genotypes.json"} if prev["e8"] else set()))
        return prev
    e1, e2, e3, e4, e6 = (load_json(f) for f in ("e1_baselines.json", "e2_scaling.json", "e3_imbalance.json", "e4_sim_circuits.json", "e6_resources.json"))
    h4, h16 = load_json("e5_hardware.json"), load_json("e5_hardware_N16.json")
    slim = lambda h: dict(N=h["N"], meta={k: h["meta"][k] for k in ("alias", "job_id", "submitted_utc", "shots", "transpiled_test", "n_qubits")},
                          summary=h["summary"], cells=h["cells"], calibration_circuits=h["calibration_circuits"], frozen_sha256=h["frozen_sha256"])
    run1 = None
    p = ROOT / "results" / "e5_hardware_run1_randomcal.json"
    if p.exists():
        r = json.loads(p.read_text())
        run1 = dict(job_id=r["meta"]["job_id"], alpha=r["summary"]["calibration"]["alpha"], err_raw=r["summary"]["mean_abs_err_raw"],
                    err_cal=r["summary"]["mean_abs_err_calibrated"], cal_ideals=[c["ideal"] for c in r["calibration_circuits"]])
    return dict(
        generated_from=[f.name for f in sorted((ROOT / "results").glob("e*.json")) if "_K" not in f.name and "screen" not in f.name and "autoM" not in f.name],
        dataset=e1["dataset"],
        e1=dict(tasks=[dict(positive=t["positive"], base_rate=t["base_rate"], logreg_all_genes=t["logreg_all_genes"],
                            by_N=[dict(N=r["N"], exact_centroid=r["exact_centroid"], logreg_p=r["logreg_p"], abstain_rate_test=r["abstain_rate_test"]) for r in t["by_N"]]) for t in e1["tasks"]]),
        e2=dict(curves=[dict(N=c["N"], C_fit=c["C_fit"], rows=c["rows"]) for c in e2["curves"]], classifier=e2["classifier"], sparsity=dict(stream_density=e2["sparsity"]["stream_density"])),
        e3=dict(N=e3["N"], synthetic=e3["synthetic"], real=e3["real"]),
        e4=[dict(N=b["N"], n_qubits=b["n_qubits"], transpiled=b["transpiled"], sketch_rel_error=b["sketch_rel_error"], test_eval_full=b["test_eval_full"],
                 cells=[dict(test_index=c["test_index"], formula=c["formula"], shots_ideal=c["shots_ideal"]) for c in b["cells"]]) for b in e4["by_N"]],
        e5=dict(N4=slim(h4), N16=slim(h16), run1=run1),
        e6=dict(ledger=e6["ledger"], assumptions=e6["assumptions"]),
        e7=compact_e7(),
        e8=compact_e8(),
    )


def main():
    tpl = (ROOT / "presentation" / "plan3_template.html").read_text()
    data = compact()
    out = tpl.replace("/*__RESULTS__*/null", json.dumps(data, separators=(",", ":")))
    assert "/*__RESULTS__*/" not in out
    dst = ROOT / "presentation" / "plan3.html"
    dst.write_text(out)
    (ROOT / "results" / "slides_data.json").write_text(json.dumps(data, indent=1))
    print(f"[slides] wrote {dst.relative_to(ROOT)} ({dst.stat().st_size/1024:.0f} kB) and results/slides_data.json")


if __name__ == "__main__":
    main()
