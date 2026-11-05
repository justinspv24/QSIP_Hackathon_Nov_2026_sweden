"""Checks that the compound-Poisson stream sampler (e13_core.Stream) has the same law as E7.poisson_stream:
per-feature mean and variance of phi over many draws, on a small random sparse matrix. Writes results/e13_selftest.json."""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e13_core as C  # noqa: E402

rng = np.random.default_rng(0)
n, N = 300, 64
P = sp.random(n, N, density=0.2, random_state=1, format="csr", dtype=np.float64)
P.data = rng.random(P.nnz)
y = (rng.random(n) < 0.1).astype(int)
w = C.class_weights(y.mean())[y]
M = 40 * N
reps = 20000
A = np.array([C.E7.poisson_stream(P, w, N, M, rng)[0] for _ in range(reps)])
st = C.Stream(P, w)
B = np.array([st.draw(M, rng)[0] for _ in range(reps)])
m_true = (w @ P.toarray()) * N / (N * n)                          # E[phi] = (N/M) * lam * sum_i w_i p_ij, lam = M/(N n)
v_true = (N / M) ** 2 * (M / (N * n)) * ((w**2) @ (P.toarray() ** 2))
out = dict(schema_version=1, experiment="e13_selftest", n=n, N=N, M=M, reps=reps,
           max_rel_mean_err_e7=float(np.max(np.abs(A.mean(0) - m_true)) / np.abs(m_true).max()),
           max_rel_mean_err_e13=float(np.max(np.abs(B.mean(0) - m_true)) / np.abs(m_true).max()),
           var_ratio_e7_median=float(np.median(A.var(0) / v_true)), var_ratio_e13_median=float(np.median(B.var(0) / v_true)),
           var_ratio_e13_over_e7_median=float(np.median(B.var(0) / A.var(0))))
zA = (A.mean(0) - m_true) / np.sqrt(v_true / reps)
zB = (B.mean(0) - m_true) / np.sqrt(v_true / reps)
out.update(z_mean_rms_e7=float(np.sqrt(np.mean(zA**2))), z_mean_rms_e13=float(np.sqrt(np.mean(zB**2))),
           z_mean_max_e13=float(np.abs(zB).max()))
out["pass_rule"] = "rms of standardised mean errors in [0.7, 1.3] (expected 1 for an unbiased sampler) and median variance ratio within 5 %"
out["pass"] = bool(0.7 < out["z_mean_rms_e13"] < 1.3 and abs(out["var_ratio_e13_median"] - 1) < 0.05)
print(json.dumps(out, indent=1))
(C.E13 / "results" / "e13_selftest.json").write_text(json.dumps(out, indent=1))
