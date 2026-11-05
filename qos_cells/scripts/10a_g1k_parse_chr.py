import sys, numpy as np, time
from cyvcf2 import VCF
c = sys.argv[1]
pan = [l.rstrip("\n").split("\t") for l in open("panel.txt")][1:]
keep = [p[0] for p in pan if p[2] in ("EUR", "EAS")]
v = VCF(f"chr{c}.vcf.gz", samples=keep, gts012=True, threads=2)
assert list(v.samples) == list(np.load("eur_eas_chr22.npz")["samples"]), "sample order mismatch"
cols, ids, t, n = [], [], time.time(), 0
for r in v:
    n += 1
    if not r.is_snp or len(r.ALT) != 1: continue
    g = r.gt_types
    if (g == 3).any(): continue
    af = g.mean() / 2
    if af < 0.05 or af > 0.95: continue
    cols.append(g.astype(np.int8)); ids.append(r.ID if r.ID else f"{c}:{r.POS}")
G = np.stack(cols, 1)
np.savez_compressed(f"eur_eas_chr{c}.npz", G=G, snp=np.array(ids))
print(f"chr{c}: variants {n:,}, kept {G.shape[1]:,}, {time.time()-t:.0f} s", flush=True)
