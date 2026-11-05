import sys, importlib.util, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hpc_common as hc
import numpy as np, cupy as cp
from pairs_gpu import PairData, hbucket, hsign
spec = importlib.util.spec_from_file_location("e7", hc.ROOT/"scripts/09_route2_classical_memory.py"); e7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e7)
from qos_cells.load_data import load_pbmc3k, select_genes, stratified_split
X, genes, labels, _ = load_pbmc3k()
y = (labels=="Dendritic cells").astype(int); tr, te = stratified_split(y, 0.3, 7)
g = select_genes(X[tr], y[tr], 512, 3)
t=time.time(); P, ab = e7.pair_features_csr(X[tr][:, g]); print("E7 csr", time.time()-t)
pd = PairData(X[tr][:, g]); 
# 1 pair norm closed form
x = X[tr][:, g].toarray().astype(np.float64); Zd = np.array([sum((x[i,a]*x[i,b])**2 for a in np.nonzero(x[i])[0] for b in np.nonzero(x[i])[0] if a<=b) for i in range(5)])
print("Z closed vs direct rel", np.abs(pd.Z[:5]-Zd).max()/Zd.max())
# 2 enumeration vs E7 csr
cid, j, p = pd.cell_pairs(np.arange(pd.n)); import scipy.sparse as sp
Q = sp.csr_matrix((cp.asnumpy(p), (cp.asnumpy(cid), cp.asnumpy(j))), shape=P.shape)
print("enum vs E7 max|diff|", abs(Q - P.astype(np.float64)).max(), "nnz", Q.nnz, P.nnz, "rowsum", Q.sum(1).max())
# 3 centroid
ytr = y[tr]; d_new = cp.asnumpy(pd.centroid_delta(np.arange(pd.n), ytr))
ind1 = (ytr==1)/ (ytr==1).sum(); ind0=(ytr==0)/(ytr==0).sum(); d_old = P.T@ind1.astype(np.float32) - P.T@ind0.astype(np.float32)
print("centroid rel diff", np.abs(d_new-d_old).max()/np.abs(d_old).max())
# 4 scores
D = np.random.default_rng(0).standard_normal(pd.N); s_new = cp.asnumpy(pd.scores(D)); s_old = P@D
print("scores rel diff", np.abs(s_new-s_old).max()/np.abs(s_old).max())
# 5 hash equality
jj = np.arange(1000, 2000); print("hash eq", (cp.asnumpy(hbucket(cp.asarray(jj), 12345, 977))==e7.hbucket(jj,12345,977)).all(), (cp.asnumpy(hsign(cp.asarray(jj), 999))==e7.hsign(jj,999)).all())
# 6 stream mean: E[phi]=Delta (in uniform mode with tau=N)
from qos_cells.stream import class_weights
w = class_weights(ytr.mean())[ytr]
t=time.time(); phi, pp, R = pd.stream(np.arange(pd.n), w, 3000*pd.N, 1); cp.cuda.Device().synchronize(); print("stream R", R, time.time()-t, "s")
ph = cp.asnumpy(phi); print("stream rel err", np.linalg.norm(ph-d_old)/np.linalg.norm(d_old), " predicted", np.sqrt(pd.N/(3000*pd.N*pd.n)*np.sum(w**2*pd.q))/np.linalg.norm(d_old))
phi7, _, _, R7 = e7.poisson_stream(P, w, pd.N, 3000*pd.N, np.random.default_rng(1)); print("E7 stream rel err", np.linalg.norm(phi7-d_old)/np.linalg.norm(d_old), "R7", R7)
