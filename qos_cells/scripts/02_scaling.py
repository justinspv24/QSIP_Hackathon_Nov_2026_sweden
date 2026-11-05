"""E2 — the N/M law. Relative error of the sketched phases vs the number of streamed records M, for several N,
measured (5 seeds) and predicted in closed form (no free parameter). Plus: classifier quality vs M at N=1024,
and the sparsity benefit of streaming only nonzero records. Writes results/e2_scaling.json."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from qos_cells import load_config
from qos_cells.load_data import load_pbmc3k, make_task
from qos_cells.stream import RecordStream
from qos_cells.sketch import sketch_phases, rescale_safe, exact_phases, scores_from_p
from qos_cells.readout import best_threshold, evaluate
from qos_cells.io import save_json


def error_curve(st, tau, M_grid, repeats, seed):
    target = st.expected_phase(tau)
    rows = []
    for M in M_grid:
        errs = []
        for r in range(repeats):
            rng = np.random.default_rng([seed, M, r])
            g, w = st.draw(M, rng)
            phi = sketch_phases(g, w, st.N, M, tau)
            errs.append(float(np.linalg.norm(phi - target) / np.linalg.norm(target)))
        rows.append(dict(M=M, rel_error_mean=float(np.mean(errs)), rel_error_std=float(np.std(errs)),
                         rel_error_predicted=st.predicted_relative_error(tau, M)))
    return rows


def main():
    cfg = load_config()
    seed, tf, mc = cfg["seed"], cfg["data"]["test_frac"], cfg["data"]["min_cells_per_gene"]
    sk, sc = cfg["sketch"], cfg["scaling"]
    X, genes, labels, summary = load_pbmc3k(cfg["data"]["path"])
    positive = sc["positive"]
    out = dict(positive=positive, stream_mode=sk["stream_mode"], curves=[], classifier=None, sparsity=None)

    # --- phase error vs M for several N (uniform stream: the paper's setting) ---
    for N in sc["gene_counts"]:
        task = make_task(X, genes, labels, positive, N, seed, tf, mc)
        st = RecordStream(task["p_train"], task["y_train"], task["pi_hat"], "uniform", task["abstain_train"])
        tau = sk["tau_per_gene"] * N
        t0 = time.time()
        rows = error_curve(st, tau, sk["M_grid"], sk["repeats"], seed)
        # fitted law: err^2 ≈ C * N / M  ->  C from the measured points
        C = float(np.mean([r["rel_error_mean"] ** 2 * r["M"] / N for r in rows]))
        out["curves"].append(dict(N=N, n_train_cells=int(st.n_cells), pi_hat=task["pi_hat"], tau=tau,
                                  stream_density=float((st.p > 0).mean()), rows=rows, C_fit=C))
        print(f"[E2] N={N:5d}: " + "  ".join(f"M={r['M']:.0e}: {r['rel_error_mean']:.3f} (pred {r['rel_error_predicted']:.3f})" for r in rows)
              + f"   err^2*M/N ≈ {C:.3f}  ({time.time()-t0:.1f}s)")

    # --- classifier quality vs M at N = classifier_N ---
    N = sc["classifier_N"]
    task = make_task(X, genes, labels, positive, N, seed, tf, mc)
    p_tr, p_te, y_tr, y_te = task["p_train"], task["p_test"], task["y_train"], task["y_test"]
    ab_tr, ab_te = task["abstain_train"], task["abstain_test"]
    st = RecordStream(p_tr, y_tr, task["pi_hat"], "uniform", ab_tr)
    tau = sk["tau_per_gene"] * N
    phi_ex, _ = exact_phases(p_tr[~ab_tr], y_tr[~ab_tr], sk["phase_limit"])
    thr_ex = best_threshold(scores_from_p(p_tr, phi_ex, ab_tr), y_tr)
    exact = evaluate(scores_from_p(p_te, phi_ex, ab_te), y_te, thr_ex)
    rows = []
    for M in sk["M_grid"]:
        ev = []
        for r in range(sk["repeats"]):
            rng = np.random.default_rng([seed, M, r, 99])
            g, w = st.draw(M, rng)
            phi, c = rescale_safe(sketch_phases(g, w, N, M, tau), sk["phase_limit"])
            thr = best_threshold(scores_from_p(p_tr, phi, ab_tr), y_tr)
            ev.append(evaluate(scores_from_p(p_te, phi, ab_te), y_te, thr))
        rows.append(dict(M=M,
                         balanced_accuracy_mean=float(np.mean([e["balanced_accuracy"] for e in ev])),
                         balanced_accuracy_std=float(np.std([e["balanced_accuracy"] for e in ev])),
                         auprc_mean=float(np.mean([e["auprc"] for e in ev])),
                         auprc_std=float(np.std([e["auprc"] for e in ev]))))
        print(f"[E2] classifier N={N} M={M:.0e}: bal.acc={rows[-1]['balanced_accuracy_mean']:.3f} AUPRC={rows[-1]['auprc_mean']:.3f}")
    print(f"[E2] exact centroid N={N}: bal.acc={exact['balanced_accuracy']:.3f} AUPRC={exact['auprc']:.3f}")
    out["classifier"] = dict(N=N, rows=rows, exact_centroid=exact, n_train_cells=int(st.n_cells))

    # --- sparsity: nonzero-only streaming needs fewer records for the same error ---
    st_nz = RecordStream(p_tr, y_tr, task["pi_hat"], "nonzero", ab_tr)
    rows_nz = error_curve(st_nz, tau, sk["M_grid"], sk["repeats"], seed)
    rows_u = [r for r in out["curves"] if r["N"] == N][0]["rows"]
    out["sparsity"] = dict(N=N, stream_density=float((st.p > 0).mean()), uniform=rows_u, nonzero=rows_nz,
                           wasted_fraction_uniform=float(1 - (st.p > 0).mean()))
    print(f"[E2] sparsity N={N}: density={out['sparsity']['stream_density']:.3f}; nonzero-mode errors: "
          + "  ".join(f"M={r['M']:.0e}: {r['rel_error_mean']:.3f}" for r in rows_nz))
    save_json("e2_scaling.json", out, cfg)


if __name__ == "__main__":
    main()
