"""Runs many e13_core jobs in parallel (each pinned to its own CPU set via taskset) and merges their JSON.

    python e13_driver.py run   --stage 1 --cores 0-15 --threads 2 --jobs-spec stage1
    python e13_driver.py merge --stage 1

Job JSON files go to logs/parts_stage{n}/ (one per job, committed only through the merged results/e13_stage{n}.json).
The merged file is rewritten after every finished job with "partial": true until all jobs are done.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e13_core as C  # noqa: E402

E13 = C.E13
PY = str(E13 / ".venv" / "bin" / "python")


def spec(stage):
    jobs = []
    if stage == 1:
        for feat, K in [("genes", 1024), ("pairs", 512), ("pairs", 2048)]:
            for t in C.TASKS["pbmc68k"]:
                jobs.append(dict(dataset="pbmc68k", feat=feat, K=K, task=t, split=0, seeds=3))
    elif stage == 2:
        for ds in ["pbmc68k", "pbmc3k"]:
            for feat, K in [("genes", 1024), ("pairs", 512), ("pairs", 2048)]:
                for split in range(5):
                    for t in C.TASKS[ds]:
                        jobs.append(dict(dataset=ds, feat=feat, K=K, task=t, split=split, seeds=5))
    # heavy jobs first
    jobs.sort(key=lambda j: -(j["K"] * (j["K"] + 1) // 2 if j["feat"] == "pairs" else j["K"]) * (30 if j["dataset"] == "pbmc68k" else 1))
    return jobs


def jname(j):
    return f"{j['dataset']}_{j['feat']}{j['K']}_{j['task'].replace('/', '_').replace(' ', '_').replace('+', 'p')}_split{j['split']}"


def cpu_list(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def run(a):
    jobs = spec(a.stage)
    parts = E13 / "logs" / f"parts_stage{a.stage}"
    parts.mkdir(parents=True, exist_ok=True)
    cpus = cpu_list(a.cores)
    slots = [cpus[i:i + a.threads] for i in range(0, len(cpus) - a.threads + 1, a.threads)]
    free = list(range(len(slots)))
    import re
    todo = [j for j in jobs if not (parts / (jname(j) + ".json")).exists() and re.search(a.filter, jname(j))
            and not (a.exclude and re.search(a.exclude, jname(j)))]
    C.log(f"stage {a.stage}: {len(jobs)} jobs, {len(todo)} to do, {len(slots)} slots x {a.threads} threads")
    running = {}
    t0 = time.time()
    while todo or running:
        while todo and free:
            j = todo.pop(0)
            lock = parts / (jname(j) + ".lock")
            if (parts / (jname(j) + ".json")).exists() or lock.exists():
                continue
            lock.write_text(str(os.getpid()))
            s = free.pop(0)
            env = dict(os.environ, OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), NUMBA_NUM_THREADS=str(a.threads))
            out = parts / (jname(j) + ".json")
            cmd = ["taskset", "-c", ",".join(map(str, slots[s])), PY, str(E13 / "scripts" / "e13_core.py"), "--dataset", j["dataset"],
                   "--feat", j["feat"], "--K", str(j["K"]), "--task", j["task"], "--split", str(j["split"]), "--seeds", str(j["seeds"]),
                   "--out", str(out)] + (["--no-logreg-full", "--rec-cap", "1e9"] if a.stage == 2 else [])
            lf = open(E13 / "logs" / f"stage{a.stage}_{jname(j)}.out", "w")
            running[subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)] = (j, s, lf, time.time())
        time.sleep(5)
        for p in [p for p in running if p.poll() is not None]:
            j, s, lf, ts = running.pop(p)
            lf.close()
            free.append(s)
            C.log(f"{jname(j)} exit {p.returncode} after {time.time() - ts:.0f} s; {len(todo)} queued, {len(running)} running")
            merge(a.stage, partial=bool(todo or running))
    C.log(f"stage {a.stage} all jobs finished in {(time.time() - t0) / 60:.1f} min")
    merge(a.stage, partial=False)


def merge(stage, partial=True):
    parts = sorted((E13 / "logs" / f"parts_stage{stage}").glob("*.json"))
    bodies = [json.loads(p.read_text()) for p in parts]
    jobs = spec(stage)
    results = [b["result"] for b in bodies]
    for b, r in zip(bodies, results):
        C.finalise_task(r)                                   # recompute every S* variant from the stored rows
        r["best_one_pass"] = C.best_one_pass(r)
        r["wall_seconds"] = b["wall_seconds"]
        r["peak_ram_gb"] = b["peak_ram_gb"]
    out = dict(schema_version=1, experiment=f"e13_stage{stage}", created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               partial=bool(partial or len(bodies) < len(jobs) or any(b.get("partial") for b in bodies)), n_jobs_done=len(bodies), n_jobs_planned=len(jobs),
               config=dict(repo=bodies[0]["config"]["repo"] if bodies else C.load_config(),
                           jobs=[b["config"]["job"] for b in bodies]),
               git_commit=C.git_commit(), job_git_commits=sorted({b["git_commit"] for b in bodies}), hardware=C.hardware(),
               wall_seconds_sum=round(sum(b["wall_seconds"] for b in bodies), 1),
               peak_ram_gb_max=max((b["peak_ram_gb"] for b in bodies), default=0), peak_vram_gb=0.0,
               dataset_summaries={b["dataset_summary"].get("dataset", "pbmc3k"): b["dataset_summary"] for b in bodies},
               tasks=results, summary_table=summary_table(results))
    if stage == 2:
        out["robustness"] = robustness(results)
    p = E13 / "results" / f"e13_stage{stage}.json"
    p.write_text(json.dumps(C.clean(out), separators=(",", ":")))
    return out


def robustness(results):
    """Per (dataset, feature set, task): S*/N per split for every method (seed-averaged rows -> stable S*), and the best
    one-pass method per split; mean and std over splits (linear) and geometric mean / std of log10 (for the figure).
    A split where a method never reaches the band counts as S*/N = 1 (store everything) in the means, and is counted."""
    import numpy as np
    groups = {}
    for r in results:
        groups.setdefault((r["dataset"], f"{r['feature_set']} K={r['K']}", r["positive"]), []).append(r)
    out = []
    for (ds, fs, pos), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["split"])
        g = dict(dataset=ds, feature_set=fs, positive=pos, N=rs[0]["N"], n_splits=len(rs), splits=[r["split"] for r in rs],
                 pi_hat=[r["pi_hat"] for r in rs], n_test_pos=[r["n_test_pos"] for r in rs],
                 exact_BA=[r["exact_centroid"]["balanced_accuracy"] for r in rs], signal=bool(np.mean([r["exact_centroid"]["balanced_accuracy"] for r in rs]) >= 0.6),
                 d_eff=[r["d_eff"] for r in rs], M_run_over_N=[r["M_run_over_N"] for r in rs])
        for var, key in [("", "S_star_over_N"), ("_Minf", "S_star_Minf_over_N")]:
            per = {}
            for k in rs[0]["methods"]:
                v = [r["methods"][k].get(key) for r in rs]
                if all(x is None for x in v) and var == "_Minf" and k == "hashing_logreg":
                    continue
                vv = np.array([x if x is not None else 1.0 for x in v])
                per[k] = dict(per_split=v, n_not_reached=int(sum(x is None for x in v)), mean=float(vv.mean()), std=float(vv.std()),
                              log10_mean=float(np.log10(vv).mean()), log10_std=float(np.log10(vv).std()))
            g["methods" + var] = per
            b = [r["best_one_pass"]["Minf" if var else "stream"]["S_star_over_N"] for r in rs]
            bb = np.array([x if x is not None else 1.0 for x in b])
            g["best_S_star_over_N_per_split" + var] = b
            g["best_methods_per_split" + var] = [r["best_one_pass"]["Minf" if var else "stream"]["method"] for r in rs]
            g["best_S_star_over_N" + var] = dict(mean=float(bb.mean()), std=float(bb.std()), log10_mean=float(np.log10(bb).mean()),
                                                  log10_std=float(np.log10(bb).std()), max=float(bb.max()), min=float(bb.min()))
        out.append(g)
    return out


def summary_table(results):
    rows = []
    for r in results:
        for k, m in r["methods"].items():
            rows.append(dict(dataset=r["dataset"], feature_set=r["feature_set"], K=r["K"], N=r["N"], positive=r["positive"], split=r["split"],
                             pi_hat=r["pi_hat"], n_train_pos=r["n_train_pos"], n_test_pos=r["n_test_pos"], tolerance=r["tolerance"],
                             exact_BA=r["exact_centroid"]["balanced_accuracy"], exact_AUPRC=r["exact_centroid"]["auprc"],
                             logreg_BA=r.get("logreg_full", {}).get("balanced_accuracy"), stream_BA=r["stream_full"]["balanced_accuracy"],
                             quantum_sim_BA=r["quantum_sim"]["balanced_accuracy"], M_run_over_N=r["M_run_over_N"],
                             M_converged_over_N=r["M_converged_over_N"], d_eff=r["d_eff"], index_qubits=r["index_qubits"],
                             method=k, S_star=m["S_star"], S_star_over_N=m["S_star_over_N"], bytes_at_S_star=m["bytes_at_S_star"],
                             S_star_tol001=m["S_star_tol001"], S_star_auprc=m["S_star_auprc"], S_star_Minf=m.get("S_star_Minf"),
                             S_star_tol002=m.get("S_star_tol002"), signal=r.get("signal")))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stage", type=int, required=True)
    r.add_argument("--cores", default="0-15")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--filter", default=".")
    r.add_argument("--exclude", default="")
    m = sub.add_parser("merge")
    m.add_argument("--stage", type=int, required=True)
    m.add_argument("--final", action="store_true")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else merge(a.stage, partial=not a.final)
