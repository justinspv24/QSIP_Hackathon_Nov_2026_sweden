"""E8 / Route 2 on a dense-signal dataset: 1000 Genomes genotypes (EUR + EAS, chromosomes 17–22).

Why this dataset
----------------
On PBMC3k (E7) the class signal Delta = mu1 - mu0 sits in a few marker features, so classical streaming sketches
with S ~ 10^-4 N counters match the exact centroid. Close human populations differ by tiny allele-frequency
shifts spread over many SNPs, so Delta should be dense (screen: results/e7b_g1k_density_screen.json). This script
asks the Route 2 question on that data, and adds the scaling test: does the classical memory S* grow in
proportion to N (N/8, N/4, N/2, N nested random SNP subsets)? The quantum register grows by log2(8) = 3 qubits.

Features: SNP dosage x in {0,1,2} (copies of the ALT allele), biallelic SNPs, MAF >= 0.05 and no missing calls in
EUR+EAS (label-free filter, computed once on all 1,007 people). Amplitude encoding as in load_data:
p = x^2 / ||x||^2 per person. No supervised feature selection at all (every common SNP is a feature).

Tasks (one vs rest, positive = the named population):
    JPT vs CHB                  Japanese vs Han Chinese (Beijing)
    JPT vs rest of East Asia    rare class, pi ~ 0.2
    FIN vs rest of Europe       rare class, pi ~ 0.2
Everything else (stream, ±1/pi weights, 5 seeds, 6 one-pass sketches + hashing+logreg, S* rules, bytes) is
run_task() from scripts/09_route2_classical_memory.py, plus one extra one-pass baseline that matters for dense but
redundant signals: random_subset = exact counters for S SNPs chosen at random before the stream (S floats + 1 seed).

Usage:  python scripts/10_route2_genotypes.py run --data <dir with eur_eas_chr*.npz> [--fractions 0.125 0.25 0.5 1]
        python scripts/10_route2_genotypes.py figure
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qos_cells import load_config  # noqa: E402
from qos_cells.load_data import stratified_split  # noqa: E402
from qos_cells.resources import ledger as qledger  # noqa: E402

_spec = importlib.util.spec_from_file_location("e7", ROOT / "scripts" / "09_route2_classical_memory.py")
e7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e7)

RESULT_NAME = "e8_route2_genotypes.json"
FIG_NAME = "fig_e8_genotype_memory.png"
TASKS = {"JPT vs CHB": ("JPT", "pop", ["JPT", "CHB"]),
         "JPT vs rest of East Asia": ("JPT", "superpop", ["EAS"]),
         "FIN vs rest of Europe": ("FIN", "superpop", ["EUR"])}


def log(m):
    print(f"[e8 {dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def load_genotypes(data_dir: Path):
    base = np.load(data_dir / "eur_eas_chr22.npz")
    pop, superpop, samples = base["pop"], base["superpop"], base["samples"]
    blocks, snps, chroms = [], [], []
    for f in sorted(data_dir.glob("eur_eas_chr*.npz")):
        d = np.load(f)
        blocks.append(d["G"]); snps.append(d["snp"]); chroms.append((f.stem.split("chr")[1], int(d["G"].shape[1])))
    G = np.concatenate(blocks, 1)
    return G, pop, superpop, samples, np.concatenate(snps), chroms


def amplitude_csr(Gsub: np.ndarray, chunk: int = 64):
    """p = x^2/||x||^2 per row, float32 csr (rows of dosages; zero dosage = zero feature)."""
    out = []
    for a in range(0, Gsub.shape[0], chunk):
        x = Gsub[a:a + chunk].astype(np.float32)
        sq = x * x
        n2 = sq.sum(1, keepdims=True)
        n2[n2 == 0] = 1.0
        out.append(sp.csr_matrix(sq / n2))
    P = sp.vstack(out).tocsr()
    P.sort_indices()
    return P


def run(args):
    cfg = load_config()
    seed = int(cfg["seed"])
    G, pop, superpop, samples, snps, chroms = load_genotypes(Path(args.data))
    Nfull = G.shape[1]
    log(f"genotypes: {G.shape[0]} people x {Nfull:,} SNPs from chromosomes {[c for c, _ in chroms]}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2026]))
    order = rng.permutation(Nfull)                       # nested subsets: the first N of one fixed permutation
    fracs = sorted(float(f) for f in args.fractions)
    route = dict(dataset="1000 Genomes phase 3 (20130502), EUR + EAS, chromosomes " + ",".join(c for c, _ in chroms),
                 snp_filter="biallelic SNP, MAF >= 0.05, no missing call, over all 1,007 EUR+EAS people (label-free)",
                 features="dosage 0/1/2, amplitude p = x^2/||x||^2, no supervised selection",
                 n_snps_total=Nfull, fractions=fracs, nested_subsets="first N SNPs of one fixed random permutation",
                 seeds=int(args.seeds), M_over_N_grid=[33, 100, 300, 900, 2700, 8100], split_seed=seed,
                 test_frac=cfg["data"]["test_frac"], hash_logreg_max_S=4096,
                 S_star_rule="stable S*: smallest S whose 5-seed mean balanced accuracy stays within tol of the exact centroid "
                             "for all larger S; tol = max(0.01, one test positive); strict-0.01 variant also stored")
    cfg["route2_genotypes"] = route
    if args.fixed_m_over_n:
        route["M_over_N_grid"] = [int(args.fixed_m_over_n)]
        route["M_policy"] = f"fixed M = {int(args.fixed_m_over_n)}·N records at every N (same records per feature, clean scaling)"
    out = dict(schema_version=1, experiment="e8_route2_genotypes", config=cfg, chromosomes=[dict(chr=c, snps=n) for c, n in chroms],
               tasks=[])
    prev = ROOT / "results" / RESULT_NAME
    if args.merge and prev.exists():
        old_json = json.loads(prev.read_text())
        out["tasks"] = [t for t in old_json["tasks"] if t["task"] not in (args.tasks or [])]
        out["config"] = old_json["config"]
        out["config"]["route2_genotypes"].setdefault("reruns", []).append(dict(tasks=args.tasks, M_over_N_grid=route["M_over_N_grid"]))
    for ti, (name, (posname, col, members)) in enumerate(TASKS.items()):
        if args.tasks and name not in args.tasks:
            continue
        lab = pop if col == "pop" else superpop
        m = np.isin(lab, members) if col == "superpop" else np.isin(pop, members)
        y = (pop[m] == posname).astype(int)
        tr, te = stratified_split(y, cfg["data"]["test_frac"], seed)
        blocks = []
        for f in fracs:
            N = int(round(f * Nfull))
            cols_idx = np.sort(order[:N])
            t = time.time()
            Gm = G[np.flatnonzero(m)][:, cols_idx]
            P = amplitude_csr(Gm)
            del Gm
            Ptr, Pte = P[tr], P[te]
            del P
            log(f"{name} N={N:,}: features built in {time.time() - t:.0f} s (train nnz {Ptr.nnz:,}, density {Ptr.nnz / (Ptr.shape[0] * N):.2f})")
            S_grid = [2**e for e in range(3, int(math.floor(math.log2(N))) + 1)]
            if S_grid[-1] != N:
                S_grid.append(N)
            r = e7.run_task(f"{name} N={N}", Ptr, y[tr], Pte, y[te], float(y[tr].mean()), N, S_grid, route["M_over_N_grid"],
                            int(args.seeds), seed * 1000 + ti * 10 + fracs.index(f), np.zeros(len(tr), bool), np.zeros(len(te), bool),
                            do_logreg=True, do_jl_dense=False, hash_logreg_max_S=4096, logreg_max_iter=200,
                            extra_methods=("random_subset",))
            r["positive"] = name
            r["fraction"] = f
            q = qledger(N, eps=cfg["resources"]["epsilon"], Q=cfg["resources"]["queries"], pi=float(y[tr].mean()),
                        gate_rate_hz=cfg["resources"]["logical_gate_rate_hz"])
            blocks.append(dict(K=None, N=N, quantum=dict(index_qubits=q["index_qubits"], logical_qubits=q["logical_qubits"],
                                                         classical_bytes_for_Delta=8 * N), real=[r], synthetic=None))
            del Ptr, Pte
            e7._finalise(blocks)
            out["tasks"] = [t_ for t_ in out["tasks"] if t_["task"] != name] + [dict(task=name, n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                                                                              by_N=blocks)]
            _dump(out)
    _dump(out, final=True)


def _scaling(out):
    """Per task and method: S*(N) table and the log-log slope of S* vs N (1 = linear growth, 0 = flat)."""
    res = {}
    for t in out["tasks"]:
        res[t["task"]] = {}
        meths = t["by_N"][0]["real"][0]["methods"].keys()
        for k in meths:
            pts = [(b["N"], b["real"][0]["methods"][k].get("S_star_cell_stable")) for b in t["by_N"]]
            good = [(n, s) for n, s in pts if s]
            slope = None
            if len(good) >= 2 and len(good) == len(pts):
                x, yv = np.log2([n for n, _ in good]), np.log2([s for _, s in good])
                slope = float(np.polyfit(x, yv, 1)[0])
            res[t["task"]][k] = dict(points=[dict(N=n, S_star=s, S_star_over_N=(s / n if s else None)) for n, s in pts],
                                     loglog_slope=slope)
    return res


def _dump(out, final=False):
    body = dict(out)
    body["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    body["partial"] = not final
    body["scaling"] = _scaling(out)
    body["summary_table"] = [row for t in out["tasks"] for row in e7._summary_table(t["by_N"])]
    body["limits"] = ["1000 Genomes population labels are self-reported ancestry of the sampled groups, not clusters of the data (no circularity)",
                      "test sets are small: ~31 positives per task; one test positive = 0.016 balanced accuracy",
                      "one split, 5 stream/hash seeds; Poissonised stream; the quantum column is a classical simulation",
                      "6 chromosomes only (17–22); nested random SNP subsets, so N/8 ... N have the same per-SNP signal",
                      "close populations: logistic regression or PCA on stored data also work; the question is only one-pass memory",
                      "a finite experiment is evidence, not proof"]
    p = ROOT / "results" / RESULT_NAME
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(e7._clean(body), separators=(",", ":")))
    log(f"wrote results/{RESULT_NAME} ({p.stat().st_size / 1024:.0f} kB){'' if final else ' [partial]'}")


def figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads((ROOT / "results" / RESULT_NAME).read_text())
    assert d["schema_version"] == 1
    INK, GOLD, RED, BLUE = "#08090B", "#D9A800", "#D94F3D", "#2F6DB5"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    style = {"count_sketch": (INK, "o-", "count-sketch"), "hashing": ("#4A4A48", "s-", "feature hashing"),
             "jl_sparse": ("#8A8984", "^-", "sparse JL"), "heavy_hitters_mg": (GOLD, "D-", "heavy hitters (MG)"),
             "heavy_hitters_top": (GOLD, "d:", "oracle top-S (upper bound)"), "reservoir": (RED, "x-", "reservoir"),
             "random_subset": (BLUE, "v-", "random SNP subset (S counters)")}
    tasks = d["tasks"]
    fig, axes = plt.subplots(2, len(tasks), figsize=(4.8 * len(tasks), 8.4), squeeze=False)
    for j, t in enumerate(tasks):
        big = t["by_N"][-1]
        tk = big["real"][0]
        ax = axes[0][j]
        for k, (col, fmt, lab) in style.items():
            m = tk["methods"].get(k)
            if not m:
                continue
            ax.errorbar([r["S"] for r in m["rows"]], [r["balanced_accuracy_mean"] for r in m["rows"]],
                        yerr=[r["balanced_accuracy_std"] for r in m["rows"]], fmt=fmt, color=col, ms=3, lw=1.2, label=lab, alpha=.9)
        ex, tol = tk["exact_centroid"]["balanced_accuracy"], tk["tolerance_cell"]
        ax.axhline(ex, color=INK, ls=":", lw=1)
        ax.axhspan(ex - tol, ex + tol, color=INK, alpha=0.07, lw=0)
        ax.axvline(big["N"], color=INK, ls="--", lw=.8)
        q = big["quantum"]["index_qubits"]
        ax.axvline(q, color=RED, ls="--", lw=1.2)
        ax.text(q * 1.15, 0.52, f"quantum:\n{q} index qubits", color=RED, fontsize=7)
        ax.text(big["N"] / 1.15, 0.52, f"S = N\n{big['N']:,}", ha="right", fontsize=7)
        ax.set_xscale("log", base=2); ax.set_ylim(0.48, 1.03)
        ax.set_xlabel("classical memory S (counters)")
        ax.set_title(f"{t['task']} · N = {big['N']:,} SNPs\nπ̂ = {tk['pi_hat']:.2f} · test positives {tk['n_test_pos']} · M = {tk['M_run_over_N']}·N", fontsize=8.5, loc="left")
        if j == 0:
            ax.set_ylabel("balanced accuracy (test, 5 seeds)")
        ax = axes[1][j]
        sc = d["scaling"][t["task"]]
        Ns = [b["N"] for b in t["by_N"]]
        for k, (col, fmt, lab) in style.items():
            if k not in sc:
                continue
            pts = sc[k]["points"]
            xs = [p["N"] for p in pts if p["S_star"]]
            ys = [p["S_star"] for p in pts if p["S_star"]]
            sl = sc[k]["loglog_slope"]
            ax.plot(xs, ys, fmt, color=col, ms=4, lw=1.3, label=f"{lab} (slope {sl:.2f})" if sl is not None else f"{lab} (some never)")
        ax.plot(Ns, Ns, "--", color=INK, lw=.8, label="S = N")
        ax.plot(Ns, [Ns[0] * 0.1 * (n / Ns[0]) for n in Ns], ":", color=INK, lw=.8, label="∝ N (slope 1)")
        ax2 = ax.twinx()
        PUR = "#7B3FA0"
        ax2.plot(Ns, [b["quantum"]["index_qubits"] for b in t["by_N"]], "P--", color=PUR, ms=5, lw=1.4)
        for n, b in zip(Ns, t["by_N"]):
            ax2.annotate(str(b["quantum"]["index_qubits"]), (n, b["quantum"]["index_qubits"]), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7, color=PUR)
        ax2.set_ylabel("quantum index qubits (purple, right axis)", color=PUR); ax2.tick_params(axis="y", colors=PUR); ax2.set_ylim(0, 25)
        ax2.spines["top"].set_visible(False)
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xlabel("feature count N (nested SNP subsets)")
        if j == 0:
            ax.set_ylabel("classical S* (counters)")
        ax.legend(fontsize=6.5, loc="upper left", framealpha=.9)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7.5, frameon=False)
    fig.suptitle("E8 · 1000 Genomes (EUR+EAS, chr17–22): top = accuracy vs memory at full N; bottom = how S* grows with N "
                 "(slope 1: memory grows like N; quantum grows like log₂N)", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    o = ROOT / "figures" / FIG_NAME
    o.parent.mkdir(exist_ok=True)
    fig.savefig(o)
    plt.close(fig)
    log(f"wrote figures/{FIG_NAME}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--fractions", nargs="+", default=[0.125, 0.25, 0.5, 1.0])
    r.add_argument("--seeds", default=5)
    r.add_argument("--tasks", nargs="+", default=None)
    r.add_argument("--fixed-m-over-n", type=float, default=None)
    r.add_argument("--merge", action="store_true")
    sub.add_parser("figure")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else figure(a)
