"""Build presentation/plan3.html from presentation/plan3_template.html + results/*.json.
The deck contains no hand-typed numbers: everything comes from the JSON files written by scripts 01–06."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qos_cells import ROOT
from qos_cells.io import load_json


def compact():
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
        generated_from=[f.name for f in sorted((ROOT / "results").glob("e*.json"))],
        dataset=e1["dataset"],
        e1=dict(tasks=[dict(positive=t["positive"], base_rate=t["base_rate"], logreg_all_genes=t["logreg_all_genes"],
                            by_N=[dict(N=r["N"], exact_centroid=r["exact_centroid"], logreg_p=r["logreg_p"], abstain_rate_test=r["abstain_rate_test"]) for r in t["by_N"]]) for t in e1["tasks"]]),
        e2=dict(curves=[dict(N=c["N"], C_fit=c["C_fit"], rows=c["rows"]) for c in e2["curves"]], classifier=e2["classifier"], sparsity=dict(stream_density=e2["sparsity"]["stream_density"])),
        e3=dict(N=e3["N"], synthetic=e3["synthetic"], real=e3["real"]),
        e4=[dict(N=b["N"], n_qubits=b["n_qubits"], transpiled=b["transpiled"], sketch_rel_error=b["sketch_rel_error"], test_eval_full=b["test_eval_full"]) for b in e4["by_N"]],
        e5=dict(N4=slim(h4), N16=slim(h16), run1=run1),
        e6=dict(ledger=e6["ledger"], assumptions=e6["assumptions"]),
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
