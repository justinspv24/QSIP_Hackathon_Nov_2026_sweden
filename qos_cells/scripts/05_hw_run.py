"""E5 — the hardware toy on IQM Resonance (inference only; the phases were trained in simulation and are frozen in
results/hw_phases.json). 8 test-cell circuits + 4 known-answer calibration circuits, 2,000 shots each.

    python scripts/05_hw_run.py              # pick a ready machine (garnet preferred), submit, save job id, wait, analyse
    python scripts/05_hw_run.py --submit     # submit only (job id -> results/hw_job.json)
    python scripts/05_hw_run.py --collect    # fetch the job in results/hw_job.json and analyse
    python scripts/05_hw_run.py --mock       # same pipeline on the free remote mock (plumbing test only: the mock
                                             # returns uniformly random bits, not simulated values)
Needs IQM_TOKEN in the environment. Writes results/hw_job.json and results/e5_hardware.json.
"""
import argparse, datetime as dt, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from qos_cells import load_config, ROOT
from qos_cells.circuits import hadamard_test_circuit, formula_z, transpile_iqm, gate_stats, run_shots
from qos_cells.readout import counts_to_z, fit_calibration, apply_calibration
from qos_cells.io import save_json
from qpu_pick import machine_states, read_token, BLOCKED, SERVER

JOB_FILE = ROOT / "results" / "hw_job.json"   # (suffix _N{N} for non-default N)


def build_circuits(fz):
    phi = np.array(fz["phi"])
    test = [hadamard_test_circuit(np.array(a), phi) for a in fz["amplitudes"]]
    cal = [hadamard_test_circuit(np.array(c["amplitudes"]), np.array(c["phi"])) for c in fz["calibration"]]
    ideal_test = [formula_z(np.array(a), phi) for a in fz["amplitudes"]]
    ideal_cal = [formula_z(np.array(c["amplitudes"]), np.array(c["phi"])) for c in fz["calibration"]]
    return test + cal, ideal_test, ideal_cal


def choose_backend(preferred, qubits, mock=False):
    from iqm.qiskit_iqm import IQMProvider
    states = {s["alias"]: s for s in machine_states(read_token())}
    if mock:
        alias = next(a for a in states if a.endswith(":mock") and states[a]["ready"] and states[a]["qubits"] >= qubits)
    else:
        ready = [a for a in preferred if a in states and states[a]["ready"] and states[a]["qubits"] >= qubits and a not in BLOCKED]
        if not ready:
            raise RuntimeError("no preferred machine is ready now: " + json.dumps({a: states[a] for a in preferred if a in states}, default=str))
        alias = ready[0]
    be = IQMProvider(SERVER, quantum_computer=alias).get_backend()
    return alias, be, states.get(alias)


def analyse(fz, counts, ideal_test, ideal_cal, meta, cfg, suffix=""):
    n_test = len(fz["amplitudes"])
    z_test = [counts_to_z(c) for c in counts[:n_test]]
    z_cal = [counts_to_z(c) for c in counts[n_test:]]
    try:
        cal = fit_calibration(ideal_cal, z_cal, cfg["hardware"]["alpha_min"])
    except RuntimeError as e:
        if not meta.get("mock"):
            raise
        print(f"[E5] (mock) {e}")
        cal = fit_calibration(ideal_cal, z_cal, alpha_min=-np.inf)
        cal["failed"] = str(e)
    z_test_cal = apply_calibration(z_test, cal) if abs(cal["alpha"]) > 1e-6 else np.array(z_test)
    thr = fz["threshold"]
    labels = np.array(fz["labels"])
    pred_raw = (np.array(z_test) > thr).astype(int)
    pred_cal = (z_test_cal > thr).astype(int)
    ideal_pred = (np.array(ideal_test) > thr).astype(int)
    shot_sigma = 1 / np.sqrt(meta["shots"])
    rows = [dict(test_index=fz["test_indices"][k], label=int(labels[k]), ideal=float(ideal_test[k]), raw=float(z_test[k]),
                 calibrated=float(z_test_cal[k]), err_raw=float(abs(z_test[k] - ideal_test[k])),
                 err_calibrated=float(abs(z_test_cal[k] - ideal_test[k])), pred_raw=int(pred_raw[k]), pred_cal=int(pred_cal[k]),
                 pred_ideal=int(ideal_pred[k])) for k in range(n_test)]
    cal_rows = [dict(kind=fz["calibration"][k]["kind"], ideal=float(ideal_cal[k]), measured=float(z_cal[k])) for k in range(len(z_cal))]
    def bal(pred):
        tpr = pred[labels == 1].mean(); tnr = 1 - pred[labels == 0].mean()
        return float((tpr + tnr) / 2)
    summary = dict(calibration=cal, threshold=thr, shot_sigma=float(shot_sigma),
                   mean_abs_err_raw=float(np.mean([r["err_raw"] for r in rows])),
                   mean_abs_err_calibrated=float(np.mean([r["err_calibrated"] for r in rows])),
                   sign_agreement_raw=float(np.mean(pred_raw == ideal_pred)), sign_agreement_calibrated=float(np.mean(pred_cal == ideal_pred)),
                   balanced_accuracy_raw=bal(pred_raw), balanced_accuracy_calibrated=bal(pred_cal), balanced_accuracy_ideal=bal(ideal_pred))
    out = dict(meta=meta, frozen_sha256=fz["sha256"], N=fz["N"], genes=fz["genes"], inference_only=True,
               cells=rows, calibration_circuits=cal_rows, summary=summary, raw_counts=[{str(k): int(v) for k, v in c.items()} for c in counts])
    name = ("e5_hardware_mock" if meta.get("mock") else "e5_hardware") + suffix + ".json"
    save_json(name, out, cfg)
    print(f"[E5] {meta['backend']}: alpha={cal['alpha']:.3f} beta={cal['beta']:+.3f} | mean|err| raw {summary['mean_abs_err_raw']:.3f} -> "
          f"calibrated {summary['mean_abs_err_calibrated']:.3f} (shot sigma {shot_sigma:.3f}) | sign agreement raw {summary['sign_agreement_raw']:.2f} "
          f"cal {summary['sign_agreement_calibrated']:.2f} | bal.acc on 8 cells: ideal {summary['balanced_accuracy_ideal']:.2f} "
          f"raw {summary['balanced_accuracy_raw']:.2f} cal {summary['balanced_accuracy_calibrated']:.2f}")
    for r in rows:
        print(f"      cell {r['test_index']:4d} y={r['label']}: ideal {r['ideal']:+.3f} raw {r['raw']:+.3f} cal {r['calibrated']:+.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--timeout", type=int, default=1500, help="seconds to wait for the job")
    ap.add_argument("--N", type=int, default=None, help="feature count (default: config hardware.N); other N use results/hw_phases_N{N}.json")
    args = ap.parse_args()
    cfg = load_config()
    hw = cfg["hardware"]
    N = args.N or hw["N"]
    fz = json.loads((ROOT / "results" / (f"hw_phases_N{N}.json" if N != hw["N"] else "hw_phases.json")).read_text())
    assert fz["N"] == N
    suffix = "" if N == hw["N"] else f"_N{N}"
    circuits, ideal_test, ideal_cal = build_circuits(fz)
    qubits = circuits[0].num_qubits
    from iqm.qiskit_iqm import IQMProvider

    job_file = JOB_FILE.with_name(f"hw_job{suffix}.json")
    if args.collect:
        meta = json.loads(job_file.read_text())
        be = IQMProvider(SERVER, quantum_computer=meta["alias"]).get_backend()
        job = be.retrieve_job(meta["job_id"])
    else:
        alias, be, state = choose_backend(hw["preferred"], qubits, mock=args.mock)
        tcs = [transpile_iqm(c, be, optimization_level=3) for c in circuits]
        stats = [gate_stats(t) for t in tcs]
        print(f"[E5] backend {alias}: {state} | transpiled test circuit: {stats[0]}")
        # sanity check against the local shot simulator before spending credits
        if not args.mock:
            sim = [counts_to_z(c) for c in run_shots(circuits, hw["shots"], cfg["seed"])]
            dev = max(abs(sim[k] - (ideal_test + ideal_cal)[k]) for k in range(len(circuits)))
            assert dev < 4 / np.sqrt(hw["shots"]), f"simulator disagrees with formula: {dev}"
        job = be.run(tcs, shots=hw["shots"])
        meta = dict(alias=alias, backend=str(be.name), job_id=str(job.job_id()), submitted_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    shots=hw["shots"], n_circuits=len(circuits), n_qubits=qubits, transpiled_test=stats[0], transpiled_calibration=stats[len(ideal_test):],
                    mock=args.mock, frozen_sha256=fz["sha256"], machine_state=state)
        job_file.write_text(json.dumps(meta, indent=1, default=str))   # saved BEFORE waiting: a crash cannot lose the job
        print(f"[E5] submitted job {meta['job_id']} to {alias}; id saved to results/{job_file.name}")
        if args.submit:
            return
    t0 = time.time()
    while True:
        st = job.status()
        print(f"[E5] job status {st} ({time.time()-t0:.0f}s)")
        if str(st).upper().endswith(("DONE", "ERROR", "CANCELLED")):
            break
        if time.time() - t0 > args.timeout:
            print("[E5] timeout: run later with --collect")
            return
        time.sleep(10)
    res = job.result()
    counts = [res.get_counts(k) for k in range(len(circuits))]
    meta = json.loads(job_file.read_text())
    meta["collected_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    analyse(fz, counts, ideal_test, ideal_cal, meta, cfg, suffix)


if __name__ == "__main__":
    main()
