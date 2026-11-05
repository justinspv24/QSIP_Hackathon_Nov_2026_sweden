"""E11 — advantage map: where do memory-S classical sketches fail, and where do real datasets sit?

Synthetic family (decision D4 in HPC/PROGRESS.md)
-------------------------------------------------
N = 2^14 ... 2^24 features. The class-mean difference Delta has k = round(N^alpha) nonzero entries (random support,
random signs, equal magnitude delta), alpha in {0, 0.25, 0.5, 0.75, 1}. For equal magnitudes the participation
ratio d_eff = ||Delta||_2^4 / ||Delta||_4^4 equals k exactly, so y = log(d_eff)/log(N) = log(k)/log(N) ~ alpha.

Training noise: the full-memory estimate is Delta_hat = Delta + tau g, g ~ N(0, I_N), tau^2 = 1/n1 + 1/n0 with
n1 = 26, n0 = 1820 (the PBMC3k dendritic-cell training split): the sampling noise of a class-mean difference of
unit-variance features.

Test cells: x = +/- Delta/2 + eps, eps ~ N(0, I_N), balanced classes. For any decoded weight vector D the score
<x, D> is Gaussian with mean +/- <Delta, D>/2 and sd ||D||, so the balanced accuracy at the midpoint threshold is
exactly   BA(D) = Phi( <Delta, D> / (2 ||D||) ).   No cells are sampled; this is what makes N = 2^24 cheap.

Fixed SNR: delta is set per (N, alpha) so that the EXPECTED exact-centroid accuracy BA(Delta_hat) is 0.95, i.e.
A / sqrt(A + N tau^2) = 2 Phi^-1(0.95) with A = ||Delta||^2 = k delta^2.

Sketches (same definitions and 64-bit mix hash as scripts/09_route2_classical_memory.py, in CuPy):
count-sketch (3 rows of S/3, median decode), signed feature hashing (1 row of S), sparse JL (4 rows into S, /4),
Misra-Gries (S/2 + S/2 counters on the positive and negative parts, mergeable rule), and oracle top-S (keep the S
largest |Delta_hat_j| exactly; an upper bound for any heavy-hitter scheme, NOT a one-pass method).
Applied to the aggregate Delta_hat: linear sketches commute with sums. For Misra-Gries the positive/negative parts
are max(Delta_hat, 0) and max(-Delta_hat, 0) (no record stream is simulated here).

S* (stable): smallest S on the grid (powers of 2 from 8 to N) such that the mean BA over 3 seeds is >= the mean
exact-centroid BA - 0.01 at this S and every larger S. Best classical one-pass S* = min over count-sketch, hashing,
sparse JL, Misra-Gries.

Usage:  python e11_advantage_map.py run | collect | figure
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpc_common import FIGURES, RESULTS, Tracker, log, write_json  # noqa: E402

NAME = "e11_advantage_map.json"
TAG = "e11"
ONE_PASS = ["count_sketch", "hashing", "jl_sparse", "heavy_hitters_mg"]
ALL_METHODS = ONE_PASS + ["heavy_hitters_top"]

CFG = dict(log2N=list(range(14, 25)), alphas=[0.0, 0.25, 0.5, 0.75, 1.0], n1=26, n0=1820, target_exact_ba=0.95,
           seeds=3, base_seed=11, S_grid="powers of 2 from 8 to N", count_sketch_rows=3, jl_sparse_rows=4,
           tolerance=0.01, S_star_rule="stable: mean BA (3 seeds) >= mean exact-centroid BA - 0.01 at S and all larger S",
           ba_formula="BA(D) = Phi(<Delta,D> / (2 ||D||)) for test x = +/-Delta/2 + N(0,I), midpoint threshold (exact)",
           noise_model="Delta_hat = Delta + tau g, tau^2 = 1/n1 + 1/n0",
           misra_gries_parts="positive part max(Delta_hat,0), negative part max(-Delta_hat,0); k = S/2 counters each",
           one_pass_methods=ONE_PASS, upper_bound_reference="heavy_hitters_top (oracle top-S, not one-pass)")

# ---- hashing (identical constants/semantics to scripts/09_route2_classical_memory.py) ---------------------------
_C1, _C2, _C3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def mix(cp, j, seed):
    x = (j ^ cp.uint64(seed)) * cp.uint64(_C1)
    x ^= x >> cp.uint64(30)
    x *= cp.uint64(_C2)
    x ^= x >> cp.uint64(27)
    x *= cp.uint64(_C3)
    x ^= x >> cp.uint64(31)
    return x


def ndtri(p):
    from scipy.special import ndtri as f
    return float(f(p))


def Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2))


def run(args):
    import cupy as cp

    tr = Tracker()
    tau2 = 1.0 / CFG["n1"] + 1.0 / CFG["n0"]
    c = 2 * ndtri(CFG["target_exact_ba"])
    results = []
    log2Ns = [int(v) for v in (args.log2N or CFG["log2N"])]
    for L in log2Ns:
        N = 2**L
        t_N = time.time()
        S_grid = [2**e for e in range(3, L + 1)]
        jj = cp.arange(N, dtype=cp.uint64)
        A = (c * c + math.sqrt(c**4 + 4 * c * c * N * tau2)) / 2      # ||Delta||^2 for expected exact BA 0.95
        # per seed: hash mixes reused across alpha (seed s -> same hash functions for every alpha)
        cfg_rows = {a: dict(exact=[], deff=None, methods={m: {S: [] for S in S_grid} for m in ALL_METHODS}) for a in CFG["alphas"]}
        for s in range(CFG["seeds"]):
            hrng = np.random.default_rng(np.random.SeedSequence([CFG["base_seed"], L, s, 1]))
            hseeds = [int(v) for v in hrng.integers(0, 2**62, size=16)]
            M = {i: mix(cp, jj, hseeds[i]) for i in range(16)}
            sign = {i: 1.0 - 2.0 * (M[i] >> cp.uint64(63)).astype(cp.float64) for i in (1, 3, 5, 7, 9, 11, 13, 15)}
            for a in CFG["alphas"]:
                k = max(1, int(round(N**a)))
                rng = cp.random.RandomState(int(np.random.SeedSequence([CFG["base_seed"], L, s, int(a * 100), 2]).generate_state(1)[0]))
                supp = rng.permutation(N)[:k] if k < N else cp.arange(N)
                delta = math.sqrt(A / k)
                Delta = cp.zeros(N)
                Delta[supp] = delta * (1.0 - 2.0 * (rng.rand(k) < 0.5))
                if s == 0:
                    n2 = float((Delta**2).sum())
                    cfg_rows[a]["deff"] = n2 * n2 / float((Delta**4).sum())
                    cfg_rows[a]["k"] = k
                    cfg_rows[a]["delta"] = delta
                Dh = Delta + math.sqrt(tau2) * rng.standard_normal(N)

                def ba(D):
                    nd = float(cp.linalg.norm(D))
                    return 0.5 if nd == 0 else Phi(float(Delta @ D) / (2 * nd))
                cfg_rows[a]["exact"].append(ba(Dh))
                pos, neg = cp.maximum(Dh, 0), cp.maximum(-Dh, 0)
                pos_sorted = cp.sort(pos)[::-1]
                neg_sorted = cp.sort(neg)[::-1]
                abs_sorted = cp.sort(cp.abs(Dh))[::-1]
                for S in S_grid:
                    m = cfg_rows[a]["methods"]
                    # count-sketch: 3 rows of B buckets, median-of-3 decode
                    B = max(1, S // 3)
                    est = []
                    for r in range(3):
                        h = (M[2 * r] % cp.uint64(B)).astype(cp.int64)
                        sg = sign[2 * r + 1]
                        C = cp.bincount(h, weights=sg * Dh, minlength=B)
                        est.append(sg * C[h])
                    e0, e1, e2 = est
                    D = cp.maximum(cp.minimum(e0, e1), cp.minimum(cp.maximum(e0, e1), e2))
                    m["count_sketch"][S].append(ba(D))
                    del est, e0, e1, e2, D
                    # feature hashing, 1 row of S
                    h = (M[6] % cp.uint64(S)).astype(cp.int64)
                    C = cp.bincount(h, weights=sign[7] * Dh, minlength=S)
                    m["hashing"][S].append(ba(sign[7] * C[h]))
                    # sparse JL: 4 rows into S counters, decode R^T R / 4
                    C = cp.zeros(S)
                    hs = []
                    for r in range(4):
                        h = (M[8 + 2 * r] % cp.uint64(S)).astype(cp.int64)
                        C += cp.bincount(h, weights=sign[9 + 2 * r] * Dh, minlength=S)
                        hs.append(h)
                    D = cp.zeros(N)
                    for r, h in enumerate(hs):
                        D += sign[9 + 2 * r] * C[h]
                    m["jl_sparse"][S].append(ba(D / 4))
                    del hs, D
                    # Misra-Gries, k = S/2 counters per sign part (mergeable rule: subtract (k+1)-th largest)
                    kk = max(1, S // 2)
                    if kk >= N:
                        D = Dh
                    else:
                        D = cp.maximum(pos - pos_sorted[kk], 0) - cp.maximum(neg - neg_sorted[kk], 0)
                    m["heavy_hitters_mg"][S].append(ba(D))
                    # oracle top-S
                    if S >= N:
                        D = Dh
                    else:
                        thr = abs_sorted[S - 1]
                        D = cp.where(cp.abs(Dh) >= thr, Dh, 0.0)
                    m["heavy_hitters_top"][S].append(ba(D))
                del Delta, Dh, pos, neg, pos_sorted, neg_sorted, abs_sorted
            del M, sign
            cp.get_default_memory_pool().free_all_blocks()
        for a in CFG["alphas"]:
            r = cfg_rows[a]
            ex = float(np.mean(r["exact"]))
            row = dict(log2N=L, N=N, alpha=a, k=r["k"], delta=r["delta"], norm2_Delta=A, tau=math.sqrt(tau2),
                       d_eff_true=r["deff"], y=math.log(r["deff"]) / math.log(N),
                       exact_ba_mean=ex, exact_ba_seeds=r["exact"], methods={})
            for mth in ALL_METHODS:
                rows = [dict(S=S, ba_mean=float(np.mean(v)), ba_std=float(np.std(v))) for S, v in r["methods"][mth].items()]
                ok = [x["ba_mean"] >= ex - CFG["tolerance"] for x in rows]
                Sst = next((rows[i]["S"] for i in range(len(rows)) if all(ok[i:])), None)
                row["methods"][mth] = dict(rows=rows, S_star=Sst, S_star_over_N=(Sst / N if Sst else None))
            cand = [(row["methods"][m]["S_star"], m) for m in ONE_PASS if row["methods"][m]["S_star"] is not None]
            if cand:
                Sb, mb = min(cand)
                row["best_one_pass"] = dict(method=mb, S_star=Sb, S_star_over_N=Sb / N)
            else:
                row["best_one_pass"] = dict(method=None, S_star=None, S_star_over_N=None)
            results.append(row)
            log(TAG, f"N=2^{L} alpha={a}: exact BA {ex:.3f}  best one-pass {row['best_one_pass']}  "
                     f"top-S S*/N {row['methods']['heavy_hitters_top']['S_star_over_N']}")
        log(TAG, f"N=2^{L} done in {time.time() - t_N:.1f} s")
        _save(results, tr, partial=(L != log2Ns[-1]))
    log(TAG, f"run done: {tr.stats()}")


def _real_points():
    pts = []
    for fn in ("e9_fullpairs.json", "e10_atlas.json"):
        p = RESULTS / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for q in d.get("deff_points", []):
            q = dict(q)
            q["source_file"] = fn
            if q.get("d_eff") and q.get("N"):
                q["y"] = math.log(q["d_eff"]) / math.log(q["N"])
                q["log2N"] = math.log2(q["N"])
            # the map covers N = 2^14 ... 2^27: gene-only spaces (N = 512) are outside it and are not placed;
            # points whose run has not finished (no S* yet) are kept out instead of being drawn as "never"
            if q["N"] < 2**14 or q.get("best_S_star_over_N") is None:
                continue
            pts.append(q)
    return pts


SHORTLAB = {"B cells": "B", "Dendritic cells": "DC", "Megakaryocytes": "Mk", "hematopoietic stem cell": "HSC",
            "plasma cell": "PC", "CD1c-positive myeloid dendritic cell": "cDC", "CD14-positive monocyte": "Mono"}


def _caption(syn, pts):
    fail = sorted({(r["alpha"]) for r in syn if (r["best_one_pass"]["S_star_over_N"] or 1.0) >= 0.25})
    easy = sorted({(r["alpha"]) for r in syn if (r["best_one_pass"]["S_star_over_N"] or 1.0) <= 1e-2})
    s = (f"Synthetic: one-pass sketches need S*/N >= 0.25 (or never reach the exact centroid) at alpha in {fail}; "
         f"S*/N <= 1e-2 at alpha in {easy} for some N (best = Misra-Gries, S* ~ d_eff). ")
    lin_never = all(r["methods"][m]["S_star"] is None for r in syn for m in ("count_sketch", "hashing", "jl_sparse"))
    if lin_never:
        s += "Linear sketches (count-sketch, hashing, sparse JL) never reach the exact centroid at any S <= N here: dense noise collides into the signal. "
    if pts:
        ys = [q["y"] for q in pts]
        ss = [q["best_S_star_over_N"] for q in pts]
        top = max(pts, key=lambda q: q["y"])
        s += (f"Real gene-pair tasks ({len(ys)}; PBMC3k K = 512…13,714, AIDA K = 512/2,048): log d_eff / log N = {min(ys):.2f}–{max(ys):.2f}; "
              f"measured best one-pass S*/N = {min(ss):.1e}–{max(ss):.1e} (all <= 1e-2). "
              f"The most spread real task ({top['label']}, y = {top['y']:.2f}) sits next to the synthetic failure edge (y ~ 0.75) "
              f"but still needs only S*/N = {top['best_S_star_over_N']:.0e}: the isotropic-noise model is pessimistic, so d_eff alone "
              "does not predict failure. No real task falls where classical sketches fail.")
    else:
        s += "Real-task points: not yet available (E9/E10 deff_points missing)."
    return s


def _save(results, tr, partial, run_resources=None):
    pts = _real_points()
    extra = {} if run_resources is None else dict(run_resources=run_resources)
    write_json(NAME, CFG, dict(synthetic=results, real_points=pts, caption=_caption(results, pts), **extra,
                               notes=["no quantum advantage is claimed; the quantum sketch would use ceil(log2 N) qubits "
                                      "but this map only measures CLASSICAL memory needs",
                                      "d_eff of the synthetic Delta equals k exactly (equal magnitudes)"]),
               tr, partial)


def collect(args):
    d = json.loads((RESULTS / NAME).read_text())
    tr = Tracker()
    # keep the GPU run's wall time / peak RAM / peak VRAM (collect itself runs on CPU only)
    run_res = d.get("run_resources") or d["resources"]
    if run_res.get("peak_vram_gb", 0) == 0:              # first collect overwrote it once: restore from the run log
        import re
        m = re.findall(r"run done: (\{.*\})", (RESULTS.parent / "logs" / "e11.log").read_text())
        if m:
            run_res = eval(m[-1])
    _save(d["synthetic"], tr, d.get("partial", False), run_resources=run_res)
    log(TAG, f"collected {len(_real_points())} real points")


def figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    d = json.loads((RESULTS / NAME).read_text())
    assert d["schema_version"] == 1
    syn, pts = d["synthetic"], d["real_points"]
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
    Ls = sorted({r["log2N"] for r in syn})
    als = sorted({r["alpha"] for r in syn})
    grid = np.full((len(als), len(Ls)), np.nan)
    for r in syn:
        v = r["best_one_pass"]["S_star_over_N"]
        grid[als.index(r["alpha"]), Ls.index(r["log2N"])] = v if v is not None else np.inf
    norm = LogNorm(vmin=1e-6, vmax=1.0)
    cmap = plt.get_cmap("viridis_r").copy()
    xe = np.r_[np.array(Ls) - 0.5, Ls[-1] + 0.5]
    ae = np.array(als)
    ye = np.r_[ae[0] - 0.125, (ae[:-1] + ae[1:]) / 2, ae[-1] + 0.125]
    finite = np.where(np.isfinite(grid), np.clip(grid, 1e-6, 1), np.nan)
    pc = ax.pcolormesh(xe, ye, np.ma.masked_invalid(finite), cmap=cmap, norm=norm, shading="flat")
    for i, a in enumerate(als):
        for j, L in enumerate(Ls):
            v = grid[i, j]
            if np.isinf(v):
                ax.add_patch(plt.Rectangle((xe[j], ye[i]), 1, ye[i + 1] - ye[i], facecolor="#D94F3D", alpha=0.85,
                                           hatch="//", edgecolor="white", lw=0))
                ax.text(L, a, "never", ha="center", va="center", fontsize=11, color="white")
            elif np.isfinite(v):
                ax.text(L, a, f"{v:.0e}".replace("e-0", "e-").replace("e+00", ""), ha="center", va="center", fontsize=10,
                        color="white" if v < 3e-3 else "black")
    cb = fig.colorbar(pc, ax=ax, pad=0.01)
    cb.set_label("best classical one-pass S* / N")
    markers = {}
    shapes = ["o", "s", "D", "^", "v", "P", "X"]
    for q in pts:
        if q.get("y") is None:
            continue
        ds = q.get("dataset", "?")
        mk = markers.setdefault(ds, shapes[len(markers) % len(shapes)])
        v = q.get("best_S_star_over_N")
        ax.scatter([q["log2N"]], [q["y"]], marker=mk, s=150, c=[v], cmap=cmap, norm=norm,
                   edgecolors="black", linewidths=1.8, zorder=5, label=ds)
        ax.annotate(SHORTLAB.get(q.get("task"), q.get("task", "")), (q["log2N"], q["y"]), xytext=(9, -16 if q.get("task") == "B cells" else -4),
                    textcoords="offset points", fontsize=10.5, ha="left", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
    h, l = ax.get_legend_handles_labels()
    if h:
        u = dict(zip(l, h))
        fig.legend(u.values(), [f"real: {k}" for k in u] , loc="lower right", bbox_to_anchor=(0.93, 0.155), ncol=len(u),
                   fontsize=11.5, frameon=False)
    xmax = max(xe[-1], max([q.get("log2N", 0) for q in pts] + [0]) + 0.8)
    ticks = list(range(Ls[0], int(math.floor(xmax)) + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(L) for L in ticks])
    ax.set_xlim(xe[0], xmax)
    ax.set_ylim(ye[0], ye[-1])
    ax.set_yticks(als)
    ax.set_xlabel("log2 N  (= index qubits of the quantum sketch)")
    ax.set_ylabel("log d_eff / log N   (signal spread; 1 = all N features)")
    ax.set_title("E11 · in the synthetic model, one-pass sketches fail once the signal spreads over ≳ N^0.75 features; "
                 "real tasks reach y = 0.74\nbut are still solved with S*/N ≤ 4e-3.\ncells: synthetic Gaussian family (exact BA 0.95), colour = best S*/N; "
                 "points: real gene-pair tasks, colour = measured S*/N", loc="left", fontsize=12.5)
    import textwrap
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(d["caption"], 165)), fontsize=11, va="bottom")
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    out = FIGURES / "fig_e11_advantage_map.png"
    fig.savefig(out)
    plt.close(fig)
    log(TAG, f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--log2N", nargs="+", type=int, default=None)
    sub.add_parser("collect")
    sub.add_parser("figure")
    a = ap.parse_args()
    {"run": run, "collect": collect, "figure": figure}[a.cmd](a)
