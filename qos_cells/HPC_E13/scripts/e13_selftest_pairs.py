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
# hashing fold == direct E7 hashing at each S
import scipy.sparse as sp
seeds = np.array([12345, 67890])
Pt = B[:300]
Hmax = C.hashed_max(Pt, seeds, 1024)
errs = []
for S in [8, 64, 1024]:
    j = np.arange(Pt.shape[1])
    Hm = sp.csr_matrix((C.E7.hsign(j, seeds[1]).astype(np.float32), (j, C.E7.hbucket(j, seeds[0], S))), shape=(Pt.shape[1], S))
    errs.append(float(np.abs(np.asarray((Pt @ Hm).todense()) - C.fold(Hmax, S)).max()))
out["hash_fold_max_abs_diff"] = max(errs)
out["pass"] = bool(out["pass"] and max(errs) < 1e-6)
print(out); (C.E13 / "results" / "e13_selftest_pairs.json").write_text(json.dumps(out, indent=1))
