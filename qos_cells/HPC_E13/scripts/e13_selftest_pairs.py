"""pair_features_fast == E7.pair_features_csr on PBMC3k (K=512, all cells). Writes results/e13_selftest_pairs.json."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e13_core as C
X, genes, labels, _ = C.load_dataset("pbmc3k")
y = (labels == "Dendritic cells").astype(int)
g = C.select_genes(X, y, 512, 3)
t = time.time(); A, ab = C.E7.pair_features_csr(X[:, g]); ta = time.time() - t
t = time.time(); B, bb = C.pair_features_fast(X[:, g]); tb = time.time() - t
out = dict(schema_version=1, experiment="e13_selftest_pairs", K=512, n=int(X.shape[0]), nnz=int(A.nnz),
           same_structure=bool((A.indptr == B.indptr).all() and (A.indices == B.indices).all() and (ab == bb).all()),
           max_abs_value_diff=float(np.abs(A.data - B.data).max()), seconds_e7=ta, seconds_fast=tb)
out["pass"] = out["same_structure"] and out["max_abs_value_diff"] < 1e-6
print(out); (C.E13 / "results" / "e13_selftest_pairs.json").write_text(json.dumps(out, indent=1))
