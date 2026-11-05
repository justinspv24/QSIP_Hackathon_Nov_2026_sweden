"""E8 — GPU circuit check: circuit-level statevector == formula-level oracle sketch, n = 10 ... 30 index qubits.

Circuit (n index qubits + 1 ancilla = top qubit; basis index = anc * 2^n + g), applied gate by gate on a complex128
CuPy statevector with in-place CUDA kernels (no full-size temporaries, so n = 30 fits: 2^31 * 16 B = 32 GiB):
  1. state preparation on the index register
       state A "uniform": H on every index qubit, starting from |0...0>   (the brief's "H on the index register")
       state B "cell"   : a sparse amplitude-encoded cell |psi> = sum_g a_g |g>, a_g >= 0, prepared by DIRECT
                          INITIALISATION of the statevector (state prep is not gate-decomposed here)
  2. H on the ancilla
  3. M controlled diagonal phase gates  V_t = |0><0| (x) I + |1><1| (x) exp(i tau w_t / M |g_t><g_t|), one per record.
     They all commute; each batch applies gates with DISTINCT targets in parallel (exact), duplicates go to later rounds.
  4. S-dagger and H on the ancilla;  <Z_anc> = P(0) - P(1)
Formula: phi = (tau/M) * bincount(g, w);   Z_formula = sum_g |a_g|^2 sin(phi_g)   (= Im <psi|U|psi>).

Caveat: the circuit is diagonal and classically easy to simulate. This is a correctness and scaling check of the
simulator against the formula, NOT evidence of quantum advantage.

Usage:  CUDA_VISIBLE_DEVICES=0 python HPC/scripts/e8_gpu_circuit.py run [--n 10 14 18 22 26 30]
        python HPC/scripts/e8_gpu_circuit.py figure
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hpc_common import FIGURES, Tracker, load_result, log, write_json  # noqa: E402

NAME = "e8_gpu_circuit.json"
CAVEAT = ("The sketch circuit is diagonal (H layer + commuting phase gates + Hadamard test) and classically easy to "
          "simulate; this is a correctness and scaling check of the simulator vs the formula, not evidence of quantum "
          "advantage.")
EXP_CFG = dict(
    n_values=[10, 14, 18, 22, 26, 30], dtype="complex128", M_records=2**22, pi_rare=0.02,
    record_model=("50 % of records on the 'cell' support (4096 genes or N), 50 % heavy-tailed over all N "
                  "(g = floor(N u^3), then a fixed random affine scramble); label y ~ Bernoulli(pi); "
                  "weight = value * (+1/pi if y else -1/(1-pi)), value ~ U(0,1)"),
    tau_rule="tau fixed from a pilot (first M/8 records): tau = (pi/2) * M / (2 * 8 * max_g |pilot sum_g|); "
             "then assert max|phi| <= pi/2 on the full stream",
    cell_support=4096, shots=[100, 1000, 10000, 100000, 1000000], shot_repeats=200, seed=8,
    state_prep_B="direct initialisation of the statevector (not gate-decomposed)",
)
TAG = "e8"


# ------------------------------------------------------------------------------------------------------------------
# kernels
# ------------------------------------------------------------------------------------------------------------------
def kernels():
    import cupy as cp
    src = r'''
    extern "C" __global__ void hgate(double2* psi, long long half, int q){
      long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
      long long stride = (long long)blockDim.x * gridDim.x;
      const double s = 0.70710678118654752440;
      long long lowmask = (1LL << q) - 1;
      for (; k < half; k += stride){
        long long i0 = ((k >> q) << (q + 1)) | (k & lowmask);
        long long i1 = i0 | (1LL << q);
        double2 a = psi[i0], b = psi[i1];
        psi[i0] = make_double2(s * (a.x + b.x), s * (a.y + b.y));
        psi[i1] = make_double2(s * (a.x - b.x), s * (a.y - b.y));
      }
    }
    extern "C" __global__ void sdg_upper(double2* psi, long long N){
      // S-dagger on the ancilla (top qubit): amplitudes with anc = 1 get a factor -i
      long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
      long long stride = (long long)blockDim.x * gridDim.x;
      for (; k < N; k += stride){
        double2 a = psi[N + k];
        psi[N + k] = make_double2(a.y, -a.x);
      }
    }
    extern "C" __global__ void cphase(double2* psi, const long long* g, const double* theta, long long n_gates, long long N){
      // controlled diagonal phase gates with DISTINCT targets g (one batch of commuting gates)
      long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
      long long stride = (long long)blockDim.x * gridDim.x;
      for (; k < n_gates; k += stride){
        long long i = N + g[k];
        double c = cos(theta[k]), s = sin(theta[k]);
        double2 a = psi[i];
        psi[i] = make_double2(c * a.x - s * a.y, s * a.x + c * a.y);
      }
    }
    '''
    mod = cp.RawModule(code=src)
    abs2sum = cp.ReductionKernel("complex128 x", "float64 y", "x.real()*x.real() + x.imag()*x.imag()",
                                 "a + b", "y = a", "0", "abs2sum")
    sinsum = cp.ReductionKernel("float64 p", "float64 y", "sin(p)", "a + b", "y = a", "0", "sinsum")
    # max | wrap( arg(b * conj(a)) - p ) |  : phase of the anc=1 half relative to the anc=0 half vs the formula phase
    phasedev = cp.ReductionKernel("complex128 a, complex128 b, float64 p", "float64 y",
                                  "fabs(remainder(arg(b * conj(a)) - p, 2.0 * M_PI))",
                                  "max(a, b)", "y = a", "0", "phasedev")
    return mod.get_function("hgate"), mod.get_function("sdg_upper"), mod.get_function("cphase"), abs2sum, sinsum, phasedev


def launch(fn, n_items, args):
    threads = 256
    blocks = int(min(max(1, (n_items + threads - 1) // threads), 65535 * 8))
    fn((blocks,), (threads,), args)


# ------------------------------------------------------------------------------------------------------------------
# one size
# ------------------------------------------------------------------------------------------------------------------
def run_size(n, cfg, K, tracker):
    import cupy as cp
    hgate, sdg, cphase, abs2sum, sinsum, phasedev = K
    dev = cp.cuda.Device()
    pool = cp.get_default_memory_pool()
    N = 1 << n
    M = int(cfg["M_records"])
    pi = float(cfg["pi_rare"])
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], n]))
    vram = dict(pool_peak=0, device_used_peak=0)

    def sample_vram():
        free, total = cp.cuda.runtime.memGetInfo()
        vram["pool_peak"] = max(vram["pool_peak"], int(pool.total_bytes()))
        vram["device_used_peak"] = max(vram["device_used_peak"], int(total - free))
        tracker.sample()

    timing = {}
    t0 = time.time()
    # ---- records (host -> device) ----------------------------------------------------------------------------
    supp_size = min(N, int(cfg["cell_support"]))
    supp = np.sort(rng.choice(N, supp_size, replace=False)).astype(np.int64)
    half = M // 2
    g_on = supp[rng.integers(0, supp_size, half)]
    u = rng.random(M - half)
    a_mul, b_add = int(rng.integers(1, N // 2 + 1)) * 2 + 1, int(rng.integers(0, N))   # odd multiplier -> bijection mod N
    g_tail = ((np.floor(N * u**3).astype(np.int64) * a_mul + b_add) % N)
    g = np.concatenate([g_on, g_tail])
    perm = rng.permutation(M)
    g = g[perm]
    y = rng.random(M) < pi
    w = rng.random(M) * np.where(y, 1.0 / pi, -1.0 / (1.0 - pi))
    # tau from a pilot of the first M/8 records
    P = M // 8
    pil = np.bincount(g[:P], weights=w[:P], minlength=0)
    tau = (math.pi / 2) * M / (2 * 8 * np.abs(pil).max())
    theta = tau * w / M
    g_d = cp.asarray(g)
    th_d = cp.asarray(theta)
    phi = cp.bincount(g_d, weights=th_d, minlength=N)               # formula-level sketch (float64, N)
    max_phi = float(cp.abs(phi).max())
    assert max_phi <= math.pi / 2 + 1e-12, f"phase wrap: max|phi| = {max_phi}"
    # batches of distinct targets: rank of each record within its gene group -> rounds
    order = cp.argsort(g_d)
    gs = g_d[order]
    pos = cp.arange(M, dtype=cp.int64)
    start_pos = cp.searchsorted(gs, gs, side="left")              # first position of each record's gene group
    rank = pos - start_pos
    order2 = cp.argsort(rank)                                       # records grouped by round
    rounds_g = gs[order2]
    rounds_th = th_d[order][order2]
    counts = cp.asnumpy(cp.bincount(rank[order2]))
    bounds = np.concatenate([[0], np.cumsum(counts)])
    del order, gs, pos, start_pos, rank, order2, g_d, th_d
    dev.synchronize()
    timing["records_and_formula_s"] = round(time.time() - t0, 3)
    sample_vram()

    out = dict(n=n, N=N, M=M, tau=tau, max_abs_phi=max_phi, n_rounds=int(len(counts)), n_distinct_targets=int(counts[0]),
               statevector_bytes=int(2 * N * 16), qubits=n + 1)
    states = {}
    for state in ("uniform", "cell"):
        st = {}
        tt = time.time()
        psi = cp.zeros(2 * N, dtype=cp.complex128)
        if state == "uniform":
            psi[0] = 1.0
            dev.synchronize()
            t = time.time()
            for q in range(n):
                launch(hgate, N, (psi, np.int64(N), np.int32(q)))
            dev.synchronize()
            st["H_layer_s"] = round(time.time() - t, 4)
            z_formula = float(sinsum(phi)) / N
        else:
            amp = np.abs(rng.standard_normal(supp_size))
            amp /= np.linalg.norm(amp)
            psi[cp.asarray(supp)] = cp.asarray(amp)
            st["H_layer_s"] = None
            z_formula = float(np.sum(amp**2 * np.sin(cp.asnumpy(phi[cp.asarray(supp)]))))
        sample_vram()
        # H on ancilla (qubit n)
        launch(hgate, N, (psi, np.int64(N), np.int32(n)))
        # controlled phase gates, round by round
        dev.synchronize()
        t = time.time()
        for r in range(len(counts)):
            a, b = int(bounds[r]), int(bounds[r + 1])
            launch(cphase, b - a, (psi, rounds_g[a:b], rounds_th[a:b], np.int64(b - a), np.int64(N)))
        dev.synchronize()
        st["phase_gates_s"] = round(time.time() - t, 4)
        sample_vram()
        # check: relative phase of the anc=1 half = phi (on the state's support)
        if state == "uniform":
            st["max_phase_dev"] = float(phasedev(psi[:N], psi[N:], phi))
        else:
            sd = cp.asarray(supp)
            st["max_phase_dev"] = float(phasedev(psi[sd], psi[N + sd], phi[sd]))
        # readout: S-dagger, H on ancilla, P0
        t = time.time()
        launch(sdg, N, (psi, np.int64(N)))
        launch(hgate, N, (psi, np.int64(N), np.int32(n)))
        p0 = float(abs2sum(psi[:N]))
        p1 = float(abs2sum(psi[N:]))
        dev.synchronize()
        st["readout_s"] = round(time.time() - t, 4)
        sample_vram()
        z_circ = p0 - p1
        st.update(z_circuit=z_circ, z_formula=z_formula, abs_err=abs(z_circ - z_formula), norm_err=abs(p0 + p1 - 1.0),
                  P0=p0, wall_s=round(time.time() - tt, 3))
        # shot noise: Binomial(shots, P0) vs the formula value
        shots = {}
        srng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], n, 77, 0 if state == "uniform" else 1]))
        p0c = min(max(p0, 0.0), 1.0)
        for s in cfg["shots"]:
            k = srng.binomial(int(s), p0c, size=int(cfg["shot_repeats"]))
            zs = 2 * k / s - 1
            sigma = math.sqrt(max(1 - z_formula**2, 1e-300) / s)
            err = np.abs(zs - z_formula)
            shots[str(s)] = dict(shots=int(s), sigma=sigma, mean_abs_err=float(err.mean()),
                                 frac_within_3sigma=float((err <= 3 * sigma).mean()),
                                 rms_err=float(np.sqrt(np.mean(err**2))))
        st["shot_noise"] = shots
        states[state] = st
        del psi
        pool.free_all_blocks()
        log(TAG, f"n={n} {state:<7} Z_circ={z_circ:+.6e} Z_formula={z_formula:+.6e} |diff|={st['abs_err']:.2e} "
                 f"phase dev={st['max_phase_dev']:.1e} wall={st['wall_s']}s")
    out["states"] = states
    out["max_abs_circuit_minus_formula"] = max(s["abs_err"] for s in states.values())
    out["max_phase_dev"] = max(s["max_phase_dev"] for s in states.values())
    out["wall_s"] = round(time.time() - t0, 3)
    out["timing"] = timing
    out["peak_vram_pool_gb"] = round(vram["pool_peak"] / 2**30, 3)
    out["peak_vram_device_gb"] = round(vram["device_used_peak"] / 2**30, 3)
    del phi, rounds_g, rounds_th
    pool.free_all_blocks()
    return out


def run(args):
    import cupy as cp
    tracker = Tracker()
    cfg = dict(EXP_CFG)
    cfg["n_values"] = [int(v) for v in args.n]
    cfg["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    K = kernels()
    # warm-up (untimed): CuPy JIT-compiles reduction kernels per launch configuration on first use (~18 s each);
    # run every kernel once on small and large arrays so compile time is not counted as simulation time
    _, _, _, abs2sum, sinsum, phasedev = K
    for e in (10, 14, 18, 22, 26):
        z = cp.ones(2**e, dtype=cp.complex128)
        f = cp.zeros(2**e)
        float(abs2sum(z)); float(sinsum(f)); float(phasedev(z, z, f))
        del z, f
    cp.get_default_memory_pool().free_all_blocks()
    log(TAG, "warm-up done")
    sizes = []
    for n in cfg["n_values"]:
        log(TAG, f"n = {n}: statevector 2^{n + 1} complex128 = {2**(n + 1) * 16 / 2**30:.3f} GiB")
        r = run_size(n, cfg, K, tracker)
        sizes.append(r)
        log(TAG, f"n={n}: max|circuit-formula| = {r['max_abs_circuit_minus_formula']:.2e}  wall {r['wall_s']} s  "
                 f"peak VRAM {r['peak_vram_device_gb']} GB")
        write_json(NAME, cfg, dict(caveat=CAVEAT, sizes=sizes, summary=summary(sizes)), tracker,
                   partial=(n != cfg["n_values"][-1]))
    log(TAG, "done")


def summary(sizes):
    worst = max(s["max_abs_circuit_minus_formula"] for s in sizes)
    shot = {}
    for s in sizes:
        for st_name, st in s["states"].items():
            for k, v in st["shot_noise"].items():
                shot.setdefault(k, []).append(v["frac_within_3sigma"])
    return dict(max_abs_circuit_minus_formula_all_sizes=worst, largest_n=max(s["n"] for s in sizes),
                frac_within_3sigma_min_by_shots={k: min(v) for k, v in shot.items()},
                verdict=(f"circuit == formula to {worst:.1e} for n up to {max(s['n'] for s in sizes)} index qubits; "
                         "shot noise matches the binomial law. Correctness/scaling check only: the circuit is diagonal "
                         "and classically easy."))


# ------------------------------------------------------------------------------------------------------------------
# figure (JSON only)
# ------------------------------------------------------------------------------------------------------------------
def figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load_result(NAME)
    S = d["sizes"]
    n = np.array([s["n"] for s in S])
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 4, figsize=(13.33, 7.5), dpi=150)
    INK, GOLD, RED, BLUE, DIM = "#08090B", "#D9A800", "#D94F3D", "#2F6DB5", "#8A8984"
    # 1 accuracy
    for st, col, mk in (("uniform", INK, "o"), ("cell", GOLD, "s")):
        e = np.array([max(s["states"][st]["abs_err"], 1e-18) for s in S])
        ax[0].semilogy(n, e, mk + "-", color=col, ms=8, lw=2, label=f"state: {st}")
    ax[0].axhline(1e-12, color=DIM, ls=":", lw=1.2)
    ax[0].set_title("|Z circuit − Z formula|", loc="left")
    ax[0].set_xlabel("index qubits n")
    ax[0].legend(fontsize=12, frameon=False)
    # 2 wall time
    for key, col, lab in (("H_layer_s", INK, "H layer (uniform)"), ("phase_gates_s", GOLD, "phase gates"),
                          ("readout_s", BLUE, "ancilla readout")):
        v = [s["states"]["uniform"][key] for s in S]
        ax[1].semilogy(n, v, "o-", color=col, ms=7, lw=2, label=lab)
    ax[1].semilogy(n, [s["wall_s"] for s in S], "k--", lw=1.5, label="total incl. setup")
    ax[1].set_title("wall time (s)", loc="left")
    ax[1].set_xlabel("index qubits n")
    ax[1].legend(fontsize=11, frameon=False)
    # 3 VRAM
    ax[2].semilogy(n, [s["peak_vram_device_gb"] for s in S], "o-", color=RED, ms=8, lw=2, label="peak VRAM (device)")
    nn = np.linspace(n.min(), n.max(), 50)
    ax[2].semilogy(nn, 2 ** (nn + 1) * 16 / 2**30, ":", color=INK, lw=1.5, label="statevector 2^(n+1)·16 B")
    ax[2].axhline(48, color=DIM, ls="--", lw=1.2)
    ax[2].text(n.min(), 52, "A6000: 48 GB", fontsize=12, color=DIM)
    ax[2].set_title("peak VRAM (GB)", loc="left")
    ax[2].set_xlabel("index qubits n")
    ax[2].legend(fontsize=11, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
    # 4 shot noise (largest n, cell state)
    big = S[-1]
    for st, col in (("uniform", INK), ("cell", GOLD)):
        sn = big["states"][st]["shot_noise"]
        sh = np.array([v["shots"] for v in sn.values()])
        ax[3].loglog(sh, [v["rms_err"] for v in sn.values()], "o-", color=col, ms=7, lw=2, label=f"RMS error, {st}")
    sh = np.array(d["config"]["experiment"]["shots"], float)
    ax[3].loglog(sh, 1 / np.sqrt(sh), "--", color=RED, lw=1.5, label="1/√shots")
    fr = d["summary"]["frac_within_3sigma_min_by_shots"]
    ax[3].set_title(f"shot noise, n = {big['n']}", loc="left")
    ax[3].set_xlabel("shots")
    ax[3].legend(fontsize=11, frameon=False)
    ax[3].text(0.02, 0.03, "within 3σ: " + f"{min(fr.values()) * 100:.0f}–{max(fr.values()) * 100:.0f} %",
               transform=ax[3].transAxes, fontsize=12)
    worst = d["summary"]["max_abs_circuit_minus_formula_all_sizes"]
    fig.suptitle(f"E8 · GPU statevector circuit = formula sketch to {worst:.0e}, up to n = {int(n.max())} index qubits "
                 f"(+1 ancilla) on one A6000\nCorrectness and scaling check only: the circuit is diagonal and classically "
                 f"easy — not evidence of quantum advantage.", fontsize=15, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.subplots_adjust(top=0.80)
    out = FIGURES / "fig_e8_scaling.png"
    fig.savefig(out)
    log(TAG, f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", nargs="+", default=EXP_CFG["n_values"])
    sub.add_parser("figure")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else figure(a)
