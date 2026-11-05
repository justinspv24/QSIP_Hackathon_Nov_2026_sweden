"""E5-IBM: an exact mirror of E5 (scripts/05_hw_run.py, IQM Garnet) on IBM Quantum, free Open Plan instance only.

Same frozen phases (results/hw_phases.json for N=4, results/hw_phases_N16.json for N=16; never recomputed, sha-checked
against the IQM result files), same 8 held-out test cells, same 5 calibration circuits (zero sketch + 4 training cells),
same 2,000 shots. All 26 circuits (13 for N=4 + 13 for N=16) go to IBM in ONE SamplerV2 job: one queue wait.

    python scripts/05b_ibm_hw_run.py                 # PREFLIGHT ONLY: account, plan check, backend, transpile stats,
                                                     # estimated QPU time, remaining free time. Submits NOTHING.
    python scripts/05b_ibm_hw_run.py --submit        # same preflight, asks you to type SUBMIT, sends one job, waits, analyses
    python scripts/05b_ibm_hw_run.py --backend NAME  # override the backend choice (still only on the Open Plan instance)
    python fetch_ibm_results.py                      # fetch the job in results/hw_job_ibm.json later and analyse

Writes results/ibm_preflight.json, results/hw_job_ibm.json, results/e5_ibm_N4.json, results/e5_ibm_N16.json,
results/e5_ibm_summary.md, then re-renders the figures and the deck (scripts/07_figures.py, scripts/08_slides.py).
The token is never printed: only saved-account NAMES are shown.
"""
import argparse, datetime as dt, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from qos_cells import load_config, ROOT

# The ONLY instance this script will ever use (free Open Plan). Hard-coded on purpose; there is no flag to change it.
INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/5ddd7611c4a647e6bb5cacfcecfd476d:2bf5a911-5445-4579-9556-32e61f40454d::"
PREFERRED = "ibm_kingston"
NS = (4, 16)
IQM_FILES = {4: "e5_hardware.json", 16: "e5_hardware_N16.json"}
PREFLIGHT = ROOT / "results" / "ibm_preflight.json"
JOB_FILE = ROOT / "results" / "hw_job_ibm.json"
SUMMARY_MD = ROOT / "results" / "e5_ibm_summary.md"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- frozen inputs (identical to E5)
def load_frozen(N, cfg):
    name = "hw_phases.json" if N == cfg["hardware"]["N"] else f"hw_phases_N{N}.json"
    fz = json.loads((ROOT / "results" / name).read_text())
    assert fz["N"] == N, f"{name}: N={fz['N']} != {N}"
    iqm = json.loads((ROOT / "results" / IQM_FILES[N]).read_text())
    assert fz["sha256"] == iqm["frozen_sha256"], f"{name} sha {fz['sha256']} != IQM run's {iqm['frozen_sha256']}: phases changed"
    assert iqm["meta"]["shots"] == cfg["hardware"]["shots"] == 2000, "shot count differs from the IQM run"
    assert len(fz["amplitudes"]) == 8 and len(fz["calibration"]) == 5, "expected 8 test cells + 5 calibration circuits"
    return fz, iqm


def build_circuits(fz):
    """Copied from scripts/05_hw_run.py: 8 test circuits then 5 calibration circuits, same order."""
    from qos_cells.circuits import hadamard_test_circuit, formula_z
    phi = np.array(fz["phi"])
    test = [hadamard_test_circuit(np.array(a), phi) for a in fz["amplitudes"]]
    cal = [hadamard_test_circuit(np.array(c["amplitudes"]), np.array(c["phi"])) for c in fz["calibration"]]
    ideal_test = [formula_z(np.array(a), phi) for a in fz["amplitudes"]]
    ideal_cal = [formula_z(np.array(c["amplitudes"]), np.array(c["phi"])) for c in fz["calibration"]]
    return test + cal, ideal_test, ideal_cal


# ---------------------------------------------------------------- IBM account / instance / backend
def connect():
    from qiskit_ibm_runtime import QiskitRuntimeService
    accts = QiskitRuntimeService.saved_accounts()          # dict name -> account; only the names are printed
    print(f"[IBM] python: {sys.executable}")
    print("[IBM] saved accounts:", ", ".join(accts.keys()) if accts else "(none)")
    if not accts:
        sys.exit("[IBM] no saved account in this Python. Run this script with the Python where you called "
                 "QiskitRuntimeService.save_account(...).")
    svc = QiskitRuntimeService(instance=INSTANCE)
    plan = "unknown"
    try:
        insts = svc.instances()
        mine = [i for i in insts if isinstance(i, dict) and i.get("crn") == INSTANCE]
        if not mine:
            sys.exit("[IBM] the Open Plan instance is not visible to this account; refusing to continue.")
        info = mine[0]
        plan = str(info.get("plan") or info.get("pricing_type") or "unknown")
        if plan != "unknown" and "open" not in plan.lower() and "free" not in plan.lower():
            sys.exit(f"[IBM] instance plan is '{plan}', not the free Open Plan; refusing to continue.")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[IBM] could not read the instance plan ({type(e).__name__}); continuing ONLY because the instance is "
              "pinned to the Open Plan CRN you gave.")
    print(f"[IBM] instance: Open Plan CRN (pinned) | plan reported: {plan}")
    return svc, plan


def median_2q_error(backend):
    t = backend.target
    errs, names = [], []
    for name in t.operation_names:
        try:
            op = t.operation_from_name(name)
        except Exception:
            continue
        if getattr(op, "num_qubits", 0) != 2:
            continue
        e = [p.error for p in (t[name] or {}).values() if p is not None and p.error is not None]
        if e:
            errs += e; names.append(name)
    return (float(np.median(errs)) if errs else None), "/".join(names)


def pick_backend(svc, need_qubits, name=None):
    bes = svc.backends(simulator=False, operational=True, min_num_qubits=need_qubits)
    queue = {}
    for b in bes:
        try:
            st = b.status(); queue[b.name] = dict(pending_jobs=int(st.pending_jobs), status_msg=str(st.status_msg))
        except Exception as e:
            queue[b.name] = dict(error=type(e).__name__)
    print("[IBM] operational backends on this instance (pending jobs):",
          ", ".join(f"{k} ({v.get('pending_jobs', '?')})" for k, v in sorted(queue.items(), key=lambda kv: kv[1].get("pending_jobs", 1e9))))
    if name:
        be = svc.backend(name); why = "requested with --backend"
    elif PREFERRED in queue:
        be = svc.backend(PREFERRED); why = "preferred (ibm_kingston is operational)"
    else:
        be = svc.least_busy(simulator=False, operational=True, min_num_qubits=need_qubits); why = "least busy (ibm_kingston not available)"
    med, gname = median_2q_error(be)
    state = dict(backend=be.name, why=why, num_qubits=int(be.num_qubits), median_2q_error=med, two_qubit_gate=gname,
                 **queue.get(be.name, {}), all_queues=queue)
    print(f"[IBM] backend: {be.name} ({be.num_qubits} qubits) — {why} | median 2-qubit ({gname}) error: "
          f"{med:.2e} | pending jobs: {state.get('pending_jobs', '?')}" if med is not None else f"[IBM] backend: {be.name} (no 2q error data)")
    return be, state


# ---------------------------------------------------------------- transpile + stats
def ibm_stats(tc):
    ops = {k: int(v) for k, v in tc.count_ops().items()}
    real = [ins for ins in tc.data if ins.operation.name not in ("barrier", "delay")]
    twoq = sum(1 for ins in real if ins.operation.num_qubits == 2)
    active = sorted({tc.find_bit(q).index for ins in real for q in ins.qubits})
    return dict(ops=ops, depth=int(tc.depth()), two_qubit=int(twoq),
                depth_2q=int(tc.depth(lambda ins: ins.operation.num_qubits == 2)),
                n_qubits=len(active), physical_qubits=active, device_qubits=int(tc.num_qubits))


def transpile_all(backend, circuits, seed):
    from qiskit.transpiler import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)
    tcs = pm.run(circuits)
    return tcs, [ibm_stats(t) for t in tcs]


def estimate_qpu_seconds(backend, tcs, shots):
    """Rough: shots x (rep delay + circuit duration) + ~2 s job overhead. IBM bills measured quantum seconds;
    the job's own estimate (job.usage_estimation) is printed right after submission."""
    t = backend.target
    def med_dur(n_q):
        d = []
        for name in t.operation_names:
            try:
                op = t.operation_from_name(name)
            except Exception:
                continue
            if getattr(op, "num_qubits", 0) == n_q or (n_q == 0 and name == "measure"):
                d += [p.duration for p in (t[name] or {}).values() if p is not None and p.duration]
        return float(np.median(d)) if d else None
    d2, dm = med_dur(2) or 1e-7, med_dur(0) or 2e-6
    rep = None
    for attr in ("default_rep_delay",):
        try:
            rep = getattr(backend, attr, None) or getattr(backend.configuration(), attr, None)
        except Exception:
            pass
    rep = float(rep) if rep else 250e-6
    if rep > 1e-2:                     # reported in microseconds
        rep *= 1e-6
    per_circ = [shots * (rep + tc.depth() * d2 + dm) for tc in tcs]
    return float(sum(per_circ) + 2.0), dict(rep_delay_s=rep, median_2q_duration_s=d2, measure_duration_s=dm)


def remaining_time(svc):
    try:
        u = svc.usage()
        keep = {k: v for k, v in (u or {}).items() if any(s in k for s in ("usage", "seconds", "limit", "remaining", "period"))}
        return keep or u
    except Exception as e:
        return dict(unavailable=f"{type(e).__name__}: {e}"[:200])


# ---------------------------------------------------------------- preflight
def preflight(args, cfg):
    shots = cfg["hardware"]["shots"]
    per_N = {}
    for N in NS:
        fz, iqm = load_frozen(N, cfg)
        circs, ideal_test, ideal_cal = build_circuits(fz)
        per_N[N] = dict(fz=fz, iqm=iqm, circuits=circs, ideal_test=ideal_test, ideal_cal=ideal_cal)
    # same sanity check as E5: local simulator vs formula before spending QPU time (skipped if qiskit-aer is missing)
    try:
        from qos_cells.circuits import run_shots
        from qos_cells.readout import counts_to_z
        for N, p in per_N.items():
            sim = [counts_to_z(c) for c in run_shots(p["circuits"], shots, cfg["seed"])]
            dev = max(abs(s - i) for s, i in zip(sim, p["ideal_test"] + p["ideal_cal"]))
            assert dev < 4 / np.sqrt(shots), f"N={N}: simulator disagrees with formula: {dev}"
        print("[IBM] local simulator agrees with the frozen formula (N=4, N=16)")
    except ImportError:
        print("[IBM] qiskit-aer not installed: skipped the local simulator sanity check")

    svc, plan = connect()
    pre_name = args.backend
    if args.submit and not pre_name and PREFLIGHT.exists():
        pre_name = json.loads(PREFLIGHT.read_text()).get("backend")      # submit to the backend you reviewed
    be, state = pick_backend(svc, need_qubits=5, name=pre_name)

    all_tcs, rows, offset = [], [], 0
    print(f"\n[IBM] transpiled with optimization_level=3 (seed {cfg['seed']}) for {be.name}")
    print(f"{'N':>3} {'#':>2} {'circuit':<13} {'qubits':>6} {'2q':>4} {'depth':>6} {'2q-depth':>8} | {'IQM 2q':>6} {'IQM depth':>9}")
    for N, p in per_N.items():
        tcs, stats = transpile_all(be, p["circuits"], cfg["seed"])
        iq = p["iqm"]["meta"]; iqm_stats = [iq["transpiled_test"]] + [None] * 7 + iq["transpiled_calibration"]
        kinds = [f"test cell {i}" for i in p["fz"]["test_indices"]] + [c["kind"] for c in p["fz"]["calibration"]]
        for k, (s, q, kind) in enumerate(zip(stats, iqm_stats, kinds)):
            print(f"{N:>3} {k:>2} {kind:<13} {s['n_qubits']:>6} {s['two_qubit']:>4} {s['depth']:>6} {s['depth_2q']:>8} | "
                  f"{(q['two_qubit'] if q else '—'):>6} {(q['depth'] if q else '—'):>9}")
        p.update(tcs=tcs, stats=stats, pub_range=[offset, offset + len(tcs)]); offset += len(tcs)
        all_tcs += tcs
        rows.append(dict(N=N, stats=stats, kinds=kinds))

    n_circ = len(all_tcs); total_shots = n_circ * shots
    est, est_detail = estimate_qpu_seconds(be, all_tcs, shots)
    rem = remaining_time(svc)
    print(f"\n[IBM] circuits: {n_circ} ({len(per_N[4]['tcs'])} for N=4 + {len(per_N[16]['tcs'])} for N=16) in ONE job")
    print(f"[IBM] shots: {shots}/circuit, {total_shots:,} total")
    print(f"[IBM] estimated QPU time (rough): {est:.0f} s  [{est_detail}]")
    print(f"[IBM] remaining free time on the instance: {rem}")
    for N in NS:
        s = per_N[N]["stats"][0]; q = per_N[N]["iqm"]["meta"]["transpiled_test"]
        print(f"[IBM] test circuit 0, N={N}: IBM {s['two_qubit']} 2q / depth {s['depth']} (2q-depth {s['depth_2q']}) on qubits "
              f"{s['physical_qubits']}  vs  IQM {q['two_qubit']} CZ / depth {q['depth']}")
    PREFLIGHT.write_text(json.dumps(dict(created_utc=now(), backend=be.name, machine_state=state, instance_plan=plan,
                                         n_circuits=n_circ, shots=shots, total_shots=total_shots, est_qpu_seconds=est,
                                         est_detail=est_detail, remaining=rem, transpiled=rows,
                                         frozen_sha256={str(N): per_N[N]["fz"]["sha256"] for N in NS}), indent=1, default=str))
    print(f"[IBM] preflight saved to results/{PREFLIGHT.name}")
    return svc, be, state, plan, per_N, all_tcs, est, rem


# ---------------------------------------------------------------- analysis (same maths and file layout as E5)
def analyse_ibm(N, fz, counts, ideal_test, ideal_cal, meta, cfg):
    from qos_cells.readout import counts_to_z, fit_calibration, apply_calibration
    from qos_cells.io import save_json
    n_test = len(fz["amplitudes"])
    z_test = [counts_to_z(c) for c in counts[:n_test]]
    z_cal = [counts_to_z(c) for c in counts[n_test:]]
    try:
        cal = fit_calibration(ideal_cal, z_cal, cfg["hardware"]["alpha_min"])
    except RuntimeError as e:          # E5 would stop here; we keep the data and flag it instead
        print(f"[E5-IBM] N={N}: {e}")
        cal = fit_calibration(ideal_cal, z_cal, alpha_min=-np.inf); cal["alpha_below_min"] = True
    z_test_cal = apply_calibration(z_test, cal) if abs(cal["alpha"]) > 1e-6 else np.array(z_test)
    thr = fz["threshold"]; labels = np.array(fz["labels"])
    pred_raw = (np.array(z_test) > thr).astype(int); pred_cal = (z_test_cal > thr).astype(int)
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
                   balanced_accuracy_raw=bal(pred_raw), balanced_accuracy_calibrated=bal(pred_cal), balanced_accuracy_ideal=bal(ideal_pred),
                   cells_correct_raw=int((pred_raw == labels).sum()), cells_correct_calibrated=int((pred_cal == labels).sum()),
                   n_cells=int(n_test))
    out = dict(meta=meta, frozen_sha256=fz["sha256"], N=fz["N"], genes=fz["genes"], inference_only=True,
               cells=rows, calibration_circuits=cal_rows, summary=summary, raw_counts=[{str(k): int(v) for k, v in c.items()} for c in counts])
    save_json(f"e5_ibm_N{N}.json", out, cfg)
    print(f"[E5-IBM] N={N} {meta['backend']}: alpha={cal['alpha']:.3f} beta={cal['beta']:+.3f} | mean|err| raw {summary['mean_abs_err_raw']:.3f}"
          f" -> calibrated {summary['mean_abs_err_calibrated']:.3f} | cells correct raw {summary['cells_correct_raw']}/8"
          f" -> calibrated {summary['cells_correct_calibrated']}/8")
    for r in rows:
        print(f"      cell {r['test_index']:4d} y={r['label']}: ideal {r['ideal']:+.3f} raw {r['raw']:+.3f} cal {r['calibrated']:+.3f}")
    return out


def write_summary():
    L = lambda f: json.loads((ROOT / "results" / f).read_text())
    q = {4: L("e5_hardware.json"), 16: L("e5_hardware_N16.json")}
    b = {4: L("e5_ibm_N4.json"), 16: L("e5_ibm_N16.json")}
    ok = lambda h, key: int(sum(c[key] == c["label"] for c in h["cells"]))
    a = lambda h: h["summary"]["calibration"]["alpha"]
    t = lambda h: h["meta"]["transpiled_test"]
    ms = b[4]["meta"]["machine_state"]
    med = ms.get("median_2q_error")
    lines = [
        f"1. Same frozen sketch, 8 test cells, 5 calibration circuits, 2,000 shots: IQM Garnet vs IBM {b[4]['meta']['backend']} "
        f"(median 2q error {med:.1e}); IBM job {b[4]['meta']['job_id']}.",
        f"2. 2-qubit gates · depth (test circuit 0): N=4 IQM {t(q[4])['two_qubit']} CZ · {t(q[4])['depth']} vs IBM {t(b[4])['two_qubit']} · {t(b[4])['depth']}; "
        f"N=16 IQM {t(q[16])['two_qubit']} CZ · {t(q[16])['depth']} vs IBM {t(b[16])['two_qubit']} · {t(b[16])['depth']}.",
        f"3. Calibration alpha (signal kept): N=4 IQM {a(q[4]):.2f} vs IBM {a(b[4]):.2f}; N=16 IQM {a(q[16]):.2f} vs IBM {a(b[16]):.2f}.",
        f"4. Cells correct raw → calibrated: N=4 IQM {ok(q[4],'pred_raw')}/8 → {ok(q[4],'pred_cal')}/8 vs IBM {ok(b[4],'pred_raw')}/8 → {ok(b[4],'pred_cal')}/8; "
        f"N=16 IQM {ok(q[16],'pred_raw')}/8 → {ok(q[16],'pred_cal')}/8 vs IBM {ok(b[16],'pred_raw')}/8 → {ok(b[16],'pred_cal')}/8.",
        f"5. Mean |error| raw → calibrated: N=4 IQM {q[4]['summary']['mean_abs_err_raw']:.3f} → {q[4]['summary']['mean_abs_err_calibrated']:.3f} vs IBM "
        f"{b[4]['summary']['mean_abs_err_raw']:.3f} → {b[4]['summary']['mean_abs_err_calibrated']:.3f}; N=16 IQM {q[16]['summary']['mean_abs_err_raw']:.3f} → "
        f"{q[16]['summary']['mean_abs_err_calibrated']:.3f} vs IBM {b[16]['summary']['mean_abs_err_raw']:.3f} → {b[16]['summary']['mean_abs_err_calibrated']:.3f}.",
    ]
    SUMMARY_MD.write_text("# E5: IQM Garnet vs IBM — summary\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print(f"[E5-IBM] summary saved to results/{SUMMARY_MD.name}")


def rerender():
    for s in ("07_figures.py", "08_slides.py"):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / s)], cwd=ROOT, env={**__import__("os").environ, "PYTHONUTF8": "1"})
        print(f"[E5-IBM] {s}: {'ok' if r.returncode == 0 else 'FAILED (exit %d)' % r.returncode}")


def collect(svc=None, timeout=0, cfg=None):
    """Fetch the job in results/hw_job_ibm.json; if done, analyse both N, write the summary and re-render."""
    cfg = cfg or load_config()
    jf = json.loads(JOB_FILE.read_text())
    if svc is None:
        svc, _ = connect()
    job = svc.job(jf["job_id"])
    t0 = time.time()
    while True:
        st = str(job.status()).upper()
        print(f"[E5-IBM] job {jf['job_id']} on {jf['backend']}: {st} ({time.time()-t0:.0f}s)")
        if st.endswith(("DONE", "ERROR", "CANCELLED")):
            break
        if time.time() - t0 >= timeout:
            print("[E5-IBM] not finished yet. Run later:  python fetch_ibm_results.py"); return None
        time.sleep(15)
    if not st.endswith("DONE"):
        sys.exit(f"[E5-IBM] job ended with status {st}: {getattr(job, 'error_message', lambda: '')()}")
    res = job.result()
    try:
        usage = dict(quantum_seconds=job.usage())
    except Exception:
        usage = {}
    outs = {}
    for N in NS:
        m = dict(jf["per_N"][str(N)])
        fz, _ = load_frozen(N, cfg)
        _, ideal_test, ideal_cal = build_circuits(fz)
        lo, hi = m["pub_range"]
        counts = [getattr(res[i].data, m["creg"]).get_counts() for i in range(lo, hi)]
        m.update(collected_utc=now(), usage=usage)
        outs[N] = analyse_ibm(N, fz, counts, ideal_test, ideal_cal, m, cfg)
    write_summary()
    rerender()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="after the preflight, ask for confirmation and submit ONE job")
    ap.add_argument("--backend", default=None, help="override backend (Open Plan instance only)")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds to wait for results after submitting")
    args = ap.parse_args()
    cfg = load_config()
    svc, be, state, plan, per_N, all_tcs, est, rem = preflight(args, cfg)
    if not args.submit:
        print("\n[IBM] PREFLIGHT ONLY — nothing was submitted. Review the numbers above, then run with --submit.")
        return
    ans = input(f"\nSubmit {len(all_tcs)} circuits x {cfg['hardware']['shots']} shots to {be.name} on the Open Plan instance "
                f"(est. {est:.0f} s QPU)? Type SUBMIT to confirm: ").strip()
    if ans != "SUBMIT":
        print("[IBM] not submitted."); return
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=be)
    job = sampler.run(all_tcs, shots=cfg["hardware"]["shots"])
    sub = now()
    try:
        ue = job.usage_estimation
    except Exception:
        ue = None
    jf = dict(job_id=job.job_id(), backend=be.name, submitted_utc=sub, instance_plan=plan, usage_estimation=ue,
              n_circuits=len(all_tcs), shots=cfg["hardware"]["shots"], per_N={})
    for N in NS:
        p = per_N[N]
        jf["per_N"][str(N)] = dict(
            alias=be.name, backend=be.name, provider="ibm_quantum_platform", instance_plan=plan, job_id=job.job_id(),
            submitted_utc=sub, shots=cfg["hardware"]["shots"], n_circuits=len(p["tcs"]), n_qubits=int(p["circuits"][0].num_qubits),
            transpiled_test=p["stats"][0], transpiled_calibration=p["stats"][len(p["ideal_test"]):], transpiled_all=p["stats"],
            mock=False, frozen_sha256=p["fz"]["sha256"], machine_state=state, pub_range=p["pub_range"],
            creg=p["circuits"][0].cregs[0].name, optimization_level=3, sampler="SamplerV2", one_job_for_N4_and_N16=True)
    JOB_FILE.write_text(json.dumps(jf, indent=1, default=str))          # saved BEFORE waiting: a crash cannot lose the job
    print(f"[IBM] submitted job {job.job_id()} to {be.name} (IBM's usage estimate: {ue}); id saved to results/{JOB_FILE.name}")
    collect(svc, timeout=args.timeout, cfg=cfg)


if __name__ == "__main__":
    main()
