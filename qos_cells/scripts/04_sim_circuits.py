"""E4 — the feasibility chain, on the same 8 test cells (4 rare, 4 common):
    exact formula  ==  statevector  ≈  2,000-shot simulator  ≈  IQM noisy fake backend  ≈  IQM remote mock (garnet:mock)
for N = 4 (3 qubits) and N = 16 (5 qubits). The phases are the frozen sketch (M = 100k records), computed
classically. Also records the transpiled gate counts for IQM's native gate set. Writes results/e4_sim_circuits.json
and results/hw_phases.json (the frozen phases + cell indices that E5 runs on hardware)."""
import hashlib, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from qos_cells import load_config, ROOT
from qos_cells.load_data import load_pbmc3k, make_task
from qos_cells.stream import RecordStream
from qos_cells.sketch import sketch_phases, rescale_safe, exact_phases, scores_from_p
from qos_cells.readout import best_threshold, counts_to_z, evaluate
from qos_cells.circuits import hadamard_test_circuit, ideal_z, formula_z, run_shots, transpile_iqm, gate_stats, calibration_phases
from qos_cells.io import save_json


def pick_cells(task, per_class):
    ok = ~task["abstain_test"]
    pos = np.flatnonzero((task["y_test"] == 1) & ok)[:per_class]
    neg = np.flatnonzero((task["y_test"] == 0) & ok)[:per_class]
    return np.r_[pos, neg]


def frozen_phases(task, cfg, M=100_000):
    sk = cfg["sketch"]
    st = RecordStream(task["p_train"], task["y_train"], task["pi_hat"], "uniform", task["abstain_train"])
    tau = sk["tau_per_gene"] * task["N"]
    rng = np.random.default_rng([cfg["seed"], task["N"], 4])
    g, w = st.draw(M, rng)
    phi_raw = sketch_phases(g, w, task["N"], M, tau)
    phi, c = rescale_safe(phi_raw, sk["phase_limit"])
    phi_ex, _ = exact_phases(task["p_train"][~task["abstain_train"]], task["y_train"][~task["abstain_train"]], sk["phase_limit"])
    rel = float(np.linalg.norm(phi_raw - st.expected_phase(tau)) / np.linalg.norm(st.expected_phase(tau)))
    thr = best_threshold(scores_from_p(task["p_train"], phi, task["abstain_train"]), task["y_train"])
    test_eval = evaluate(scores_from_p(task["p_test"], phi, task["abstain_test"]), task["y_test"], thr)
    return dict(phi=phi, phi_exact=phi_ex, M=M, tau=tau, scale_c=c, rel_error_vs_target=rel, threshold=thr, test_eval=test_eval)


def main():
    cfg = load_config()
    cc = cfg["circuits"]
    seed = cfg["seed"]
    X, genes, labels, _ = load_pbmc3k(cfg["data"]["path"])
    from iqm.qiskit_iqm import IQMFakeApollo
    fake = IQMFakeApollo()        # 20-qubit square lattice (Garnet topology) with IQM's published noise figures
    out = dict(positive=cc["positive"], shots=cc["shots"], by_N=[])

    for N in cc["gene_counts"]:
        task = make_task(X, genes, labels, cc["positive"], N, seed, cfg["data"]["test_frac"], cfg["data"]["min_cells_per_gene"])
        fz = frozen_phases(task, cfg)
        cells = pick_cells(task, cc["cells_per_class"])
        phi = fz["phi"]
        circuits = [hadamard_test_circuit(task["x_test"][i], phi, measure=True) for i in cells]
        t0 = time.time()
        counts_ideal = run_shots(circuits, cc["shots"], seed)
        counts_noisy = run_shots(circuits, cc["shots"], seed, noise_backend=fake)
        tc = transpile_iqm(circuits[0], fake)
        stats = gate_stats(tc)
        rows = []
        for k, i in enumerate(cells):
            a = task["x_test"][i]
            rows.append(dict(test_index=int(i), label=int(task["y_test"][i]), formula=formula_z(a, phi), statevector=ideal_z(a, phi),
                             shots_ideal=counts_to_z(counts_ideal[k]), shots_fake_noisy=counts_to_z(counts_noisy[k])))
        # remote free mock (garnet:mock) if a token is present
        mock_info = None
        if os.environ.get("IQM_TOKEN"):
            try:
                from iqm.qiskit_iqm import IQMProvider
                be = IQMProvider("https://resonance.iqm.tech", quantum_computer=cc["mock_alias"]).get_backend()
                tcs = [transpile_iqm(c, be) for c in circuits]
                job = be.run(tcs, shots=cc["shots"])
                res = job.result()
                for k in range(len(cells)):
                    rows[k]["shots_remote_mock"] = counts_to_z(res.get_counts(k))
                mock_info = dict(alias=cc["mock_alias"], job_id=str(job.job_id()), backend=str(be.name))
                print(f"[E4] N={N}: remote mock {cc['mock_alias']} job {job.job_id()} done")
            except Exception as e:  # the mock is optional: never let it break the chain
                mock_info = dict(alias=cc["mock_alias"], error=repr(e)[:300])
                print(f"[E4] N={N}: remote mock failed: {e!r}"[:300])
        sv_err = max(abs(r["formula"] - r["statevector"]) for r in rows)
        sh_err = max(abs(r["formula"] - r["shots_ideal"]) for r in rows)
        print(f"[E4] N={N:2d} ({int(np.log2(N))+1} qubits): sketch rel.err {fz['rel_error_vs_target']:.3f} | test bal.acc {fz['test_eval']['balanced_accuracy']:.3f} "
              f"AUPRC {fz['test_eval']['auprc']:.3f} | |formula-statevector| max {sv_err:.1e} | |formula-shots| max {sh_err:.3f} | "
              f"transpiled: {stats['ops']} depth {stats['depth']} ({time.time()-t0:.1f}s)")
        for r in rows:
            print(f"      cell {r['test_index']:4d} y={r['label']}: formula {r['formula']:+.3f} sv {r['statevector']:+.3f} "
                  f"shots {r['shots_ideal']:+.3f} fake-noisy {r['shots_fake_noisy']:+.3f} mock {r.get('shots_remote_mock', float('nan')):+.3f}")
        out["by_N"].append(dict(N=N, n_qubits=int(np.log2(N)) + 1, genes=task["genes"], threshold=fz["threshold"],
                                sketch_M=fz["M"], sketch_rel_error=fz["rel_error_vs_target"], scale_c=fz["scale_c"],
                                test_eval_full=fz["test_eval"], phi=phi, phi_exact=fz["phi_exact"], cells=rows,
                                transpiled=stats, remote_mock=mock_info))
        if True:  # frozen phases for every N (results/hw_phases_N{N}.json; hw_phases.json = the hardware N)
            # Known-answer calibration circuits of the SAME family as the test circuits: the frozen oracle applied to
            # TRAINING cells (2 rare, 2 common, never test cells), whose ideal scores span the full range, plus the
            # zero-sketch circuit (ideal exactly 0) that anchors the offset. Same gate structure -> same shrinkage.
            ok = ~task["abstain_train"]
            n_half = cfg["hardware"]["n_calibration"] // 2
            tr_pos = np.flatnonzero((task["y_train"] == 1) & ok)[:n_half]
            tr_neg = np.flatnonzero((task["y_train"] == 0) & ok)[:n_half]
            calibration = [dict(kind="zero", amplitudes=task["x_test"][cells[0]].tolist(), phi=[0.0] * N)]
            calibration += [dict(kind="train_rare" if task["y_train"][i] == 1 else "train_common", train_index=int(i),
                                 amplitudes=task["x_train"][i].tolist(), phi=phi.tolist()) for i in np.r_[tr_pos, tr_neg]]
            payload = dict(N=N, positive=cc["positive"], phi=phi.tolist(), threshold=fz["threshold"], genes=task["genes"],
                           test_indices=[int(i) for i in cells], labels=[int(task["y_test"][i]) for i in cells],
                           amplitudes=[task["x_test"][i].tolist() for i in cells],
                           calibration=calibration, sketch_M=fz["M"], scale_c=fz["scale_c"])
            payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
            (ROOT / "results" / f"hw_phases_N{N}.json").write_text(json.dumps(payload, indent=1))
            if N == cfg["hardware"]["N"]:
                (ROOT / "results" / "hw_phases.json").write_text(json.dumps(payload, indent=1))
            print(f"[E4] frozen phases written: results/hw_phases_N{N}.json (sha {payload['sha256']})")
    save_json("e4_sim_circuits.json", out, cfg)


if __name__ == "__main__":
    main()
