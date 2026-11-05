"""E3 — the cost of rare classes. Sketch error vs M for class fractions pi in {0.5, 0.1, 0.02, 0.005} on a
controlled synthetic pool (same two expression profiles, only pi changes), then on the real rare classes.
Overlay: closed-form prediction (no free parameter) and the simple law  samples ∝ 1/(pi (1 - pi)).
Writes results/e3_imbalance.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from qos_cells import load_config
from qos_cells.load_data import load_pbmc3k, make_task
from qos_cells.stream import RecordStream, synthetic_profiles, synthetic_pool
from qos_cells.sketch import sketch_phases
from qos_cells.io import save_json


def error_curve(st, tau, M_grid, repeats, seed):
    target = st.expected_phase(tau)
    rows = []
    for M in M_grid:
        errs = []
        for r in range(repeats):
            rng = np.random.default_rng([seed, M, r, 3])
            g, w = st.draw(M, rng)
            phi = sketch_phases(g, w, st.N, M, tau)
            errs.append(float(np.linalg.norm(phi - target) / np.linalg.norm(target)))
        rows.append(dict(M=M, rel_error_mean=float(np.mean(errs)), rel_error_std=float(np.std(errs)),
                         rel_error_predicted=st.predicted_relative_error(tau, M)))
    # err^2 = C / M  ->  records needed for rel. error 0.1
    C = float(np.mean([r["rel_error_mean"] ** 2 * r["M"] for r in rows]))
    C_pred = float(np.mean([r["rel_error_predicted"] ** 2 * r["M"] for r in rows]))
    return rows, C / 0.01, C_pred / 0.01


def main():
    cfg = load_config()
    seed, sk, im = cfg["seed"], cfg["sketch"], cfg["imbalance"]
    N, tau = im["N"], sk["tau_per_gene"] * im["N"]
    out = dict(N=N, synthetic=[], real=[], law="samples for fixed error ∝ 1/(pi(1-pi))")

    # --- synthetic: only pi changes ---
    c0, c1 = synthetic_profiles(N, np.random.default_rng([seed, 11]))
    n_total = 2 * im["synthetic_cells_per_class"]
    for pi in im["pis"]:
        p, y, pi_real = synthetic_pool(c0, c1, n_total, pi, np.random.default_rng([seed, 12, int(pi * 1e4)]))
        st = RecordStream(p, y, pi_real, "uniform")
        rows, M01, M01_pred = error_curve(st, tau, im["M_grid"], im["repeats"], seed)
        out["synthetic"].append(dict(pi=pi, pi_realised=pi_real, n_pos=int(y.sum()), n_cells=int(len(y)),
                                     rows=rows, M_for_err_0p1=M01, M_for_err_0p1_predicted=M01_pred,
                                     law_factor=1 / (pi_real * (1 - pi_real))))
        print(f"[E3] synthetic pi={pi:<6}: " + "  ".join(f"M={r['M']:.0e}: {r['rel_error_mean']:.3f} (pred {r['rel_error_predicted']:.3f})" for r in rows)
              + f"  | M(err=0.1) ≈ {M01:.2e} (pred {M01_pred:.2e})")
    ref = out["synthetic"][0]
    for s in out["synthetic"]:
        s["cost_ratio_vs_balanced_measured"] = s["M_for_err_0p1"] / ref["M_for_err_0p1"]
        s["cost_ratio_vs_balanced_law"] = s["law_factor"] / ref["law_factor"]
        print(f"[E3]   pi={s['pi']:<6} cost vs balanced: measured x{s['cost_ratio_vs_balanced_measured']:.1f}, law x{s['cost_ratio_vs_balanced_law']:.1f}")

    # --- real rare classes (pi fixed by biology) ---
    X, genes, labels, _ = load_pbmc3k(cfg["data"]["path"])
    for positive in cfg["tasks"]["positives"]:
        task = make_task(X, genes, labels, positive, N, seed, cfg["data"]["test_frac"], cfg["data"]["min_cells_per_gene"])
        st = RecordStream(task["p_train"], task["y_train"], task["pi_hat"], "uniform", task["abstain_train"])
        rows, M01, M01_pred = error_curve(st, tau, im["M_grid"], im["repeats"], seed)
        out["real"].append(dict(positive=positive, pi_hat=task["pi_hat"], n_train_pos=task["n_train_pos"], rows=rows,
                                M_for_err_0p1=M01, M_for_err_0p1_predicted=M01_pred,
                                law_factor=1 / (task["pi_hat"] * (1 - task["pi_hat"]))))
        print(f"[E3] real {positive:16s} pi={task['pi_hat']:.4f}: M(err=0.1) ≈ {M01:.2e} (pred {M01_pred:.2e}), law factor {out['real'][-1]['law_factor']:.1f}")
    save_json("e3_imbalance.json", out, cfg)


if __name__ == "__main__":
    main()
